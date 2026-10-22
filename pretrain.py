# Give both teams a head start by copying the scripted bots (imitation learning, also called behaviour cloning),
# before self-play takes over. A network trained from scratch has to stumble on "walk to a site together, stand
# still for 5 steps, then hold it" by trying things at random, which can take millions of steps (or never happen).
# Copying the bots first gives it that plan; self-play then improves on it. AlphaStar and OpenAI Five started the same way.
#
#   caffeinate -i python pretrain.py                 1. plays bot v bot games, recording what each player saw and did
#                                                    2. trains one network per team to pick the bot's action (and to
#                                                       predict how the round goes, for the critic)
#                                                    3. checks the copies against the bots, saves runs/pretrain/bc.pt
#
# Then set START_FROM in config.py to runs/pretrain/bc.pt for both teams and run train.py as usual.
# Rewards and rules are untouched: only the networks' starting point changes.
#
# To make the copies robust, 10% of the moves during data collection are random (NOISE), but the network is always
# taught the bot's intended action, so it also learns how to recover from mistakes.

import os
import sys
import time
import numpy as np
import torch

import config as cfg
from env import GameEnv, TEAMS
from bots import Bots
from model import make_net, get_device, OBS_KEYS
from matches import play_match

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 600
NOISE = 0.1             # share of executed moves that are random (labels stay the bot's choice)
EPOCHS = 8
BATCH = 64              # pieces of SEQ_LEN steps per gradient step
LR = 5e-4
OUT_DIR = os.path.join(cfg.RUN_DIR, "pretrain")
device = get_device()
rng = np.random.default_rng(cfg.SEED + 7)
torch.manual_seed(cfg.SEED + 7)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- 1. record bot v bot games ----------
# One sequence per (game, team, player): what it saw, what the bot chose, and the rewards that followed.
env = GameEnv(cfg.MAP_FILE)
bots = Bots(env, seed=cfg.SEED + 7)
sequences = {team: [] for team in TEAMS}
t0 = time.time()
wins = 0
for g in range(GAMES):
    obs = env.reset(seed=100_000 + g)
    bots.reset()
    seq = {team: [{"obs": [], "action": [], "reward": [], "valid": []} for _ in range(env.n[team])] for team in TEAMS}
    done = False
    while not done:
        acts, labels = {}, {}
        for team in TEAMS:
            labels[team] = bots.actions(team)
            acts[team] = labels[team].copy()
            for i in range(env.n[team]):
                if rng.random() < NOISE:
                    acts[team][i] = [rng.integers(cfg.N_MOVES), rng.integers(cfg.N_TURNS), rng.integers(cfg.N_SPEEDS)]
        setup = env.t < cfg.SETUP_STEPS
        alive = {team: env.alive[team].copy() for team in TEAMS}
        new_obs, rewards, done, info = env.step(acts)
        for team in TEAMS:
            for i in range(env.n[team]):
                s = seq[team][i]
                s["obs"].append(obs[team][i])
                s["action"].append(labels[team][i])
                s["reward"].append(rewards[team][i])
                # learn from steps where the player is in and its action matters (attackers are frozen in setup)
                s["valid"].append(float(alive[team][i] and not (team == "attackers" and setup)))
        obs = new_obs
    wins += info["winner"] == "attackers"
    for team in TEAMS:
        sequences[team] += seq[team]
    if (g + 1) % 100 == 0:
        print(f"recorded {g + 1} games ({time.time() - t0:.0f}s), attackers won {wins / (g + 1):.0%}")


def pieces(team):
    # cut every sequence into pieces of SEQ_LEN steps (the last one padded and masked), with the discounted return
    # from each step to the end of the round as the critic's target
    L = cfg.SEQ_LEN
    out = {key: [] for key in OBS_KEYS + ["action", "ret", "valid"]}
    for s in sequences[team]:
        n = len(s["action"])
        ret = np.zeros(n, dtype=np.float32)
        running = 0.0
        for t in reversed(range(n)):
            running = s["reward"][t] + cfg.GAMMA * running
            ret[t] = running
        for start in range(0, n, L):
            idx = list(range(start, min(start + L, n)))
            pad = L - len(idx)
            for key in OBS_KEYS:
                x = np.stack([s["obs"][t][key] for t in idx])
                out[key].append(np.concatenate([x, np.zeros((pad,) + x.shape[1:], dtype=np.float32)]))
            out["action"].append(np.concatenate([np.array([s["action"][t] for t in idx]), np.zeros((pad, 3), dtype=np.int64)]))
            out["ret"].append(np.concatenate([ret[idx], np.zeros(pad, dtype=np.float32)]))
            out["valid"].append(np.concatenate([np.array([s["valid"][t] for t in idx], dtype=np.float32), np.zeros(pad, dtype=np.float32)]))
    # (pieces, L, ...) -> (L, pieces, ...), the layout evaluate_sequence expects
    return {key: torch.as_tensor(np.stack(v).swapaxes(0, 1)) for key, v in out.items()}


# ---------- 2. train one network per team to copy its bots ----------
nets = {}
for team in TEAMS:
    data = pieces(team)
    n_pieces = data["action"].shape[1]
    net = make_net(team).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    print(f"\n{team}: {n_pieces} pieces of {cfg.SEQ_LEN} steps, {int(data['valid'].sum()):,} steps to learn from")
    for epoch in range(EPOCHS):
        perm = torch.randperm(n_pieces)
        losses, correct, counted = [], np.zeros(3), 0.0
        for k in range(0, n_pieces, BATCH):
            idx = perm[k:k + BATCH]
            batch = {key: data[key][:, idx].to(device) for key in OBS_KEYS}
            actions = data["action"][:, idx].to(device)
            valid = data["valid"][:, idx].to(device)
            h0 = net.initial_state(len(idx), device)
            starts = torch.zeros(valid.shape, device=device)
            # same pass as the PPO update: log-probability of the bot's actions and the critic's value
            T, B = actions.shape[:2]
            x = net.encode({key: v.reshape((T * B,) + v.shape[2:]) for key, v in batch.items()}).reshape(T, B, -1)
            h, outs = h0, []
            for t in range(T):
                h = net.gru(x[t], h)
                outs.append(x[t] + net.mem_out(h))
            dists, value = net.heads(torch.stack(outs).reshape(T * B, -1))
            flat_a = actions.reshape(T * B, -1)
            logp = sum(d.log_prob(flat_a[:, j]) for j, d in enumerate(dists)).reshape(T, B)
            nv = valid.sum().clamp(min=1.0)
            bc_loss = -(logp * valid).sum() / nv
            v_loss = 0.5 * (((value.reshape(T, B) - data["ret"][:, idx].to(device)) ** 2) * valid).sum() / nv
            loss = bc_loss + 0.1 * v_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.MAX_GRAD_NORM)
            opt.step()
            with torch.no_grad():
                losses.append(bc_loss.item())
                for j, d in enumerate(dists):
                    hit = (d.probs.argmax(dim=1) == flat_a[:, j]).float().reshape(T, B)
                    correct[j] += (hit * valid).sum().item()
                counted += valid.sum().item()
        acc = correct / counted
        print(f"  epoch {epoch + 1}: copy loss {np.mean(losses):.3f}, picks the bot's move {acc[0]:.0%}, turn {acc[1]:.0%}, "
              f"walk/run switch {acc[2]:.0%}")
    nets[team] = net.eval()

# ---------- 3. check the copies against the bots, and save ----------
print("\nhow the copies play (100 games each):")
eval_env = GameEnv(cfg.MAP_FILE)
eval_bots = Bots(eval_env)
for a_pol, d_pol in [("bot", "bot"), ("trained", "bot"), ("bot", "trained"), ("trained", "trained"), ("trained", "random")]:
    r = play_match(eval_env, eval_bots, {"attackers": a_pol, "defenders": d_pol}, nets, 100, device, seed=60_000)
    name = {"trained": "copied", "bot": "bot", "random": "random"}
    print(f"  {name[a_pol]:>6} attackers v {name[d_pol]:<6} defenders: attackers win {r['attacker_win']:.0%}, "
          f"planted {r['planted']:.0%}, defused {r['defused']:.0%}, takedowns by A {r['outs_by_attackers']:.2f} by D {r['outs_by_defenders']:.2f}")
ckpt = {team: nets[team].state_dict() for team in TEAMS}
ckpt.update({"games": 0, "steps": 0, "game_version": cfg.GAME_VERSION, "pretrained_on_bot_games": GAMES})
path = os.path.join(OUT_DIR, "bc.pt")
torch.save(ckpt, path)
print(f"\nsaved {path}")
