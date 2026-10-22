# PPO self-play training for the v4 attack / defend game: attackers and defenders learn at the same time.
#
# Usage:
#   caffeinate -i python train.py                    full run (TOTAL_STEPS from config)
#   caffeinate -i python train.py smoke 20000        quick check that everything runs (name, steps)
#
# Output goes to runs/<run name>/:
#   run_card.md      every setting used, the game version and the git commit
#   metrics.csv      one row per update (win rates, plants, eliminations, losses, ...)
#   curves.png       plots of metrics.csv, refreshed every recording
#   bots.csv         trained teams against the scripted bots at every recording
#   gifs/            one recorded game every RECORD_EVERY games
#   checkpoints/     network weights saved with each GIF, plus latest.pt
#
# The loop, in short:
#   1. play N_ENVS games in parallel (worker processes) for ROLLOUT_STEPS steps, storing what happened
#   2. work out for every step how much better or worse things went than expected (advantages, GAE)
#   3. nudge each team's network towards actions that went better than expected (PPO update)
#   4. repeat
# Opponents: in each game a team plays as its current self, as a past version (opponent pool), or is played by the
# scripted bots (BOT_TEAM_PROB). Only current selves learn.

import os
import sys
import csv
import copy
import time
import datetime
from glob import glob
import subprocess
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from env import GameEnv, TEAMS
from workers import ParallelGames
from model import make_net, get_device, OBS_KEYS, ACTION_PARTS, obs_to_tensors
from render import draw_frame, save_gif
from bots import Bots
from matches import play_match

E, T = cfg.N_ENVS, cfg.ROLLOUT_STEPS
N_AGENTS = {"attackers": cfg.N_ATTACKERS, "defenders": cfg.N_DEFENDERS}
total_steps = int(sys.argv[2]) if len(sys.argv) > 2 else cfg.TOTAL_STEPS

np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
device = get_device()


# ======================================================================
# run folder and run card
# ======================================================================
map_name = os.path.splitext(os.path.basename(cfg.MAP_FILE))[0]
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
default_name = f"{cfg.GAME_VERSION}_{map_name}_{cfg.N_ATTACKERS}v{cfg.N_DEFENDERS}_{stamp}"
run_name = sys.argv[1] if len(sys.argv) > 1 else default_name
run_dir = os.path.join(cfg.RUN_DIR, run_name)
os.makedirs(os.path.join(run_dir, "gifs"), exist_ok=True)
os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)


def git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]).decode().strip()
        return commit + (" (plus uncommitted changes)" if dirty else "")
    except Exception:
        return "unknown (not a git repo)"


settings = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
with open(os.path.join(run_dir, "run_card.md"), "w") as f:
    f.write(f"# Run card: {run_name}\n\n")
    f.write(f"- started: {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
    f.write(f"- game version: {cfg.GAME_VERSION} (see formulation.md)\n")
    f.write(f"- git commit: {git_commit()}\n")
    f.write(f"- device: {device}, {cfg.N_WORKERS} game workers\n")
    f.write(f"- map: {cfg.MAP_FILE}, teams: {cfg.N_ATTACKERS} attackers v {cfg.N_DEFENDERS} defenders\n")
    f.write(f"- out after {cfg.WATCH_STEPS} steps in an enemy's sights (within {cfg.WATCH_ANGLE} degrees); "
            f"plant {cfg.PLANT_STEPS} steps, spike {cfg.SPIKE_STEPS}, defuse {cfg.DEFUSE_STEPS}\n")
    f.write(f"- rewards (attackers; defenders get minus): out {cfg.REWARD_OUT:+}, "
            f"plant {cfg.REWARD_PLANT:+}, win {cfg.REWARD_WIN:+}, site shaping {cfg.SITE_SHAPING} (potential-based)\n")
    f.write(f"- learning: {', '.join(cfg.TRAIN_TEAMS)}; memory: {', '.join(cfg.MEMORY_TEAMS)}; "
            f"bots play one team in {cfg.BOT_TEAM_PROB:.0%} of games; discount {cfg.GAMMA}\n")
    for team, path in cfg.START_FROM.items():
        f.write(f"- {team} start from {path} (and keep learning)\n")
    f.write(f"- total steps: {total_steps:,}\n\n")
    f.write("## All settings\n\n")
    for k, v in settings.items():
        f.write(f"- `{k}` = {v!r}\n")
print(f"run: {run_dir}   device: {device}")


# ======================================================================
# games, networks, optimisers
# ======================================================================
# the E training games run in parallel worker processes (see workers.py), one per CPU core
games = ParallelGames(cfg.MAP_FILE, [cfg.SEED + i for i in range(E)], N_AGENTS, cfg.N_WORKERS)
obs = games.reset()
# what this process needs to know about each game before stepping it (the games themselves live in the workers)
game_t = np.zeros(E, dtype=int)                                                   # step count
game_alive = {team: np.ones((E, N_AGENTS[team]), dtype=bool) for team in TEAMS}  # who is still in
eval_env = GameEnv(cfg.MAP_FILE)

nets = {team: make_net(team).to(device) for team in TEAMS}
# teams that start from a saved network (START_FROM) instead of from scratch
for team, path in cfg.START_FROM.items():
    nets[team].load_state_dict(torch.load(path, map_location=device, weights_only=False)[team])
    print(f"{team} start from {path}")
opts = {team: torch.optim.Adam(nets[team].parameters(), lr=cfg.LEARNING_RATE, eps=1e-5) for team in cfg.TRAIN_TEAMS}

# ---------- opponent pool ----------
# pool[team] holds frozen snapshots of that team's network from earlier in training.
# past_nets[team] is the snapshot that team plays with, in games where it plays as a past version.
pool = {team: [copy.deepcopy(nets[team].state_dict())] for team in cfg.TRAIN_TEAMS}
snapshot_every = cfg.SNAPSHOT_EVERY     # doubles each time the pool fills up (see the main loop)
past_nets = {team: make_net(team).to(device) for team in TEAMS}
pool_rng = np.random.default_rng(cfg.SEED + 1)

# ---------- teams that do not learn ----------
# They play with the networks an earlier run saved: its final network in normal games, and up to POOL_SIZE of its
# checkpoints, spread from the start of that run (weak) to the end (strong), as their opponent pool.
FROZEN_TEAMS = [team for team in TEAMS if team not in cfg.TRAIN_TEAMS]
if FROZEN_TEAMS:
    frozen_ckpts = sorted(glob(os.path.join(cfg.FROZEN_RUN, "checkpoints", "ckpt_*.pt")))
    picks = np.unique(np.linspace(0, len(frozen_ckpts) - 1, min(cfg.POOL_SIZE, len(frozen_ckpts))).round().astype(int))
    latest = torch.load(os.path.join(cfg.FROZEN_RUN, "checkpoints", "latest.pt"), map_location=device, weights_only=False)
    loaded = [torch.load(frozen_ckpts[i], map_location=device, weights_only=False) for i in picks]
    for team in FROZEN_TEAMS:
        nets[team].load_state_dict(latest[team])
        nets[team].eval()
        pool[team] = [ck[team] for ck in loaded]
    print(f"frozen {', '.join(FROZEN_TEAMS)} from {cfg.FROZEN_RUN}")
POOL_TEAMS = FROZEN_TEAMS if FROZEN_TEAMS else TEAMS


def pick_opponents():
    # for a new game: (team playing as a past version or None, team played by the bots or None).
    # Bots and past versions only ever stand in for one team, so the other team learns from the game.
    if pool_rng.random() < cfg.BOT_TEAM_PROB:
        candidates = [t for t in TEAMS if [o for o in TEAMS if o != t][0] in cfg.TRAIN_TEAMS]
        return None, str(pool_rng.choice(candidates))
    if cfg.OPPONENT_POOL and pool_rng.random() < cfg.POOL_PROB:
        return str(pool_rng.choice(POOL_TEAMS)), None
    return None, None


def load_past_opponents():
    for team in TEAMS:
        past_nets[team].load_state_dict(pool[team][pool_rng.integers(len(pool[team]))])


past_team, bot_team = map(list, zip(*[pick_opponents() for _ in range(E)]))

# ---------- benchmark bots ----------
bot_env = GameEnv(cfg.MAP_FILE)
bots = Bots(bot_env)

# reference numbers for this map and team size (attackers' win rate), measured once as yardsticks
REF = {}
if cfg.BOT_EVAL_GAMES > 0:
    print(f"measuring bot reference numbers for {cfg.N_ATTACKERS}v{cfg.N_DEFENDERS} on {cfg.MAP_FILE}...")
    for a_pol, d_pol, key in [("random", "random", "random_vs_random"), ("bot", "bot", "bots_vs_bots"),
                              ("bot", "random", "bots_vs_random"), ("random", "bot", "random_vs_bots")]:
        REF[key] = play_match(bot_env, bots, {"attackers": a_pol, "defenders": d_pol}, None, 300, device, seed=50_000)["attacker_win"]
        print(f"  {a_pol} attackers v {d_pol} defenders: attackers win {REF[key]:.0%}")
    with open(os.path.join(run_dir, "run_card.md"), "a") as f:
        f.write("\n## Bot reference numbers (attackers' win rate)\n\n")
        for k, val in REF.items():
            f.write(f"- {k}: {val:.1%}\n")


def stack_team(obs_all, team):
    # every player of one team across all parallel games -> dict of arrays of shape (E * n, ...)
    flat = [o[team][k] for o in obs_all for k in range(len(o[team]))]
    return {key: np.stack([x[key] for x in flat]) for key in OBS_KEYS}


def to_device(arrays):
    return {key: torch.as_tensor(x, device=device) for key, x in arrays.items()}


# the shape of each part of one player's observation, e.g. {"rays": (31, 4), ...}
OBS_SHAPES = {key: x.shape for key, x in obs[0]["attackers"][0].items()}


# storage for one rollout, per team: index [step, game, player]
def new_buffer(n, memory):
    extra = {"h": np.zeros((T, E, n, cfg.HIDDEN), dtype=np.float32)} if memory else {}   # memory going into each step
    return {
        **extra,
        **{key: np.zeros((T, E, n) + OBS_SHAPES[key], dtype=np.float32) for key in OBS_KEYS},
        "action": np.zeros((T, E, n, ACTION_PARTS), dtype=np.int64),     # (move, turn, switch) per player
        "logp": np.zeros((T, E, n), dtype=np.float32),
        "value": np.zeros((T, E, n), dtype=np.float32),
        "reward": np.zeros((T, E, n), dtype=np.float32),
        "done": np.zeros((T, E, n), dtype=np.float32),
        # valid = this step counts for learning. Not valid: attackers during the setup phase (they are frozen),
        # players who are out, and a team played by a past version or by the bots.
        "valid": np.zeros((T, E, n), dtype=np.float32),
    }


buf = {team: new_buffer(N_AGENTS[team], nets[team].memory) for team in TEAMS}
round_end = np.zeros((T, E), dtype=np.float32)    # 1 where a game's round ended at that step (memory is wiped after it)

# each player's memory during play (None for a team without memory), for the current and the past networks
hidden = {team: nets[team].initial_state(E * N_AGENTS[team], device) for team in TEAMS}
past_hidden = {team: past_nets[team].initial_state(E * N_AGENTS[team], device) for team in TEAMS}


# ======================================================================
# recording: one game as a GIF + checkpoint, bot benchmarks, and the curves plot
# ======================================================================
def record(games_done, steps_done):
    o = eval_env.reset(seed=cfg.FIXED_EVAL_SEED)
    frames = [draw_frame(eval_env)]
    done = False
    h = {team: nets[team].initial_state(len(o[team]), device) for team in TEAMS}
    while not done:
        acts = {}
        for team in TEAMS:
            with torch.no_grad():
                a, _, _, h[team] = nets[team].act(to_device(stack_team([o], team)), h[team])
            acts[team] = a.cpu().numpy()
        o, _, done, info = eval_env.step(acts)
        frames.append(draw_frame(eval_env))
    save_gif(frames, os.path.join(run_dir, "gifs", f"game_{games_done:07d}.gif"))

    ckpt = {team: nets[team].state_dict() for team in TEAMS}
    ckpt.update({"games": games_done, "steps": steps_done, "game_version": cfg.GAME_VERSION, "settings": settings})
    torch.save(ckpt, os.path.join(run_dir, "checkpoints", f"ckpt_{games_done:07d}.pt"))
    torch.save(ckpt, os.path.join(run_dir, "checkpoints", "latest.pt"))

    # benchmark against the fixed bots: unlike self-play, these opponents never change
    if cfg.BOT_EVAL_GAMES > 0:
        att = play_match(bot_env, bots, {"attackers": "trained", "defenders": "bot"}, nets, cfg.BOT_EVAL_GAMES, device)
        dfn = play_match(bot_env, bots, {"attackers": "bot", "defenders": "trained"}, nets, cfg.BOT_EVAL_GAMES, device)
        with open(bots_csv, "a", newline="") as f:
            csv.writer(f).writerow([steps_done, games_done, att["attacker_win"], dfn["attacker_win"]])
        print(f"  -> vs bots: trained attackers beat bot defenders {att['attacker_win']:.0%} | "
              f"bot attackers beat trained defenders {dfn['attacker_win']:.0%} (bots v bots {REF['bots_vs_bots']:.0%})")
    return info["winner"]


def plot_curves():
    path = os.path.join(run_dir, "metrics.csv")
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size < 2:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    x = data["steps"]
    ax = axes[0, 0]
    ax.plot(x, data["attacker_win_rate"], color="#c0392b", label="attackers win (self-play)")
    ax.plot(x, data["plant_rate"], color="#e0b000", label="spike planted")
    ax.plot(x, data["defuse_rate"], color="#2563b0", label="spike defused")
    ax.set_title("self-play rounds")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    ax.plot(x, data["outs_per_round"], color="#7a5c00")
    ax.set_title("players taken out per round (self-play)")
    ax = axes[1, 0]
    ax.plot(x, data["avg_round_length"], color="#555555")
    ax.set_title("round length (steps)")
    ax = axes[1, 1]
    ax.plot(x, data["attacker_entropy"], color="#c0392b", label="attackers")
    ax.plot(x, data["defender_entropy"], color="#2563b0", label="defenders")
    ax.set_title("policy entropy (high = random, low = decisive)")
    ax.legend()

    # benchmark panel
    ax = axes[0, 2]
    if os.path.exists(bots_csv):
        bd = np.atleast_1d(np.genfromtxt(bots_csv, delimiter=",", names=True))
        if bd.size >= 1:
            ax.plot(bd["steps"], bd["trained_attackers_vs_bot_defenders"], "o-", color="#c0392b",
                    label="trained attackers beat bot defenders (higher = better attackers)")
            ax.plot(bd["steps"], bd["bot_attackers_vs_trained_defenders"], "o-", color="#2563b0",
                    label="bot attackers beat trained defenders (lower = better defenders)")
    for key, label in [("bots_vs_bots", "bots v bots"), ("bots_vs_random", "bot attackers v random"),
                       ("random_vs_bots", "random attackers v bot defenders")]:
        if key in REF:
            ax.axhline(REF[key], color="grey", linestyle=":", linewidth=1)
            ax.text(0, REF[key] + 0.015, f"{label} ({REF[key]:.0%})", fontsize=7, color="grey")
    ax.set_title("vs fixed benchmark bots (attackers' win rate)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper right")
    axes[1, 2].axis("off")
    axes[1, 2].text(0, 0.9, "How to read this", fontsize=10, fontweight="bold")
    axes[1, 2].text(0, 0.05,
                    "Self-play numbers can go up and down forever,\n"
                    "because both teams keep changing.\n\n"
                    "The bot panel uses opponents that never change.\n"
                    "Red going up: attackers getting better.\n"
                    "Blue going down: defenders getting better.\n"
                    "Grey lines are measured at the start of each run.",
                    fontsize=9, va="bottom")
    for ax in axes.flat[:5]:
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "curves.png"))
    plt.close(fig)


# ======================================================================
# PPO update for one team
# ======================================================================
def ppo_update(team, b, last_value):
    # --- advantages with GAE ---
    # advantage = "how much better did things go than the critic expected", smoothed over time.
    # Worked backwards from the last step. done cuts the chain (no future after the round or after being taken out).
    adv = np.zeros((T, E, N_AGENTS[team]), dtype=np.float32)
    carry = np.zeros((E, N_AGENTS[team]), dtype=np.float32)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else b["value"][t + 1]
        not_done = 1.0 - b["done"][t]
        delta = b["reward"][t] + cfg.GAMMA * next_value * not_done - b["value"][t]
        carry = delta + cfg.GAMMA * cfg.GAE_LAMBDA * not_done * carry
        carry = carry * b["valid"][t]       # steps that don't count don't pass anything back
        adv[t] = carry
    returns = adv + b["value"]              # target for the critic

    if nets[team].memory:
        return ppo_update_sequences(team, b, adv, returns)

    # --- keep only the steps that count, as flat arrays ---
    keep = b["valid"].reshape(-1) > 0
    n = int(keep.sum())
    if n < 2:
        return {}
    to_t = lambda x, dtype=torch.float32: torch.as_tensor(x.reshape(T * E * N_AGENTS[team], *x.shape[3:])[keep], dtype=dtype, device=device)
    o = {key: to_t(b[key]) for key in OBS_KEYS}
    actions = to_t(b["action"], torch.int64)
    old_logp, adv_t, ret_t = to_t(b["logp"]), to_t(adv), to_t(returns)

    # normalise advantages once over the whole batch (keeps step sizes sensible).
    # Done here, not per minibatch: a tiny minibatch has no meaningful spread and gives NaN.
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    # --- PPO: several passes over the data in minibatches ---
    net, opt = nets[team], opts[team]
    stats = {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    for epoch in range(cfg.PPO_EPOCHS):
        perm = torch.randperm(n, device=device)
        # exactly MINIBATCHES chunks of (nearly) equal size, so no tiny leftover chunk
        for idx in torch.tensor_split(perm, cfg.MINIBATCHES):
            new_logp, entropy, value = net.evaluate({key: x[idx] for key, x in o.items()}, actions[idx])
            entropy = entropy.mean()
            a = adv_t[idx]

            # ratio = how much more (or less) likely the new policy makes this action than before
            ratio = torch.exp(new_logp - old_logp[idx])
            # clipped objective: take the improvement, but never move further than CLIP_EPS in one go
            pg_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.CLIP_EPS, 1 + cfg.CLIP_EPS) * a).mean()
            v_loss = 0.5 * ((value - ret_t[idx]) ** 2).mean()
            loss = pg_loss + cfg.VF_COEF * v_loss - cfg.ENT_COEF * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.MAX_GRAD_NORM)
            opt.step()

            with torch.no_grad():
                stats["pg_loss"].append(pg_loss.item())
                stats["v_loss"].append(v_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(((ratio - 1) - torch.log(ratio)).mean().item())
                stats["clip_frac"].append(((ratio - 1).abs() > cfg.CLIP_EPS).float().mean().item())
    return {k: float(np.mean(vals)) for k, vals in stats.items()}


def ppo_update_sequences(team, b, adv, returns):
    # The PPO update for a team with memory. Instead of shuffling single steps, it learns from whole sequences
    # (one per game and player: the rollout in order), replaying the memory from the state it had at the start of
    # the rollout and wiping it wherever a round ended, exactly as during play. Steps that don't count are masked out.
    # The rollout is cut into pieces of SEQ_LEN steps; each piece starts from the memory stored for its first step.
    n_ag = N_AGENTS[team]
    L = cfg.SEQ_LEN
    C = T // L
    S = C * E * n_ag                     # number of pieces
    assert C * L == T, "ROLLOUT_STEPS must be a multiple of SEQ_LEN"

    def seq(x, dtype=torch.float32):
        # (T, E, n, ...) -> (L, pieces, ...): piece c of player s holds steps c*L to c*L + L - 1
        x = x.reshape((C, L, E * n_ag) + x.shape[3:]).swapaxes(0, 1)
        return torch.as_tensor(np.ascontiguousarray(x).reshape((L, S) + x.shape[3:]), dtype=dtype, device=device)

    o = {key: seq(b[key]) for key in OBS_KEYS}
    actions = seq(b["action"], torch.int64)
    old_logp, adv_t, ret_t, valid = seq(b["logp"]), seq(adv), seq(returns), seq(b["valid"])
    h0 = seq(b["h"])[0]
    # wipe the memory where a new round began inside a piece (the stored memory at a piece's start is already wiped)
    starts = np.zeros((T, E), dtype=np.float32)
    starts[1:] = round_end[:-1]
    starts[::L] = 0.0
    starts = seq(np.repeat(starts[:, :, None], n_ag, axis=2))
    n_valid = valid.sum()
    if n_valid < 2:
        return {}
    # normalise advantages once over the steps that count
    mean = (adv_t * valid).sum() / n_valid
    std = torch.sqrt((((adv_t - mean) * valid) ** 2).sum() / n_valid)
    adv_t = (adv_t - mean) / (std + 1e-8)

    net, opt = nets[team], opts[team]
    stats = {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    for epoch in range(cfg.PPO_EPOCHS):
        perm = torch.randperm(S, device=device)
        for idx in torch.tensor_split(perm, cfg.MINIBATCHES):
            new_logp, entropy, value = net.evaluate_sequence({key: x[:, idx] for key, x in o.items()}, actions[:, idx],
                                                             h0[idx], starts[:, idx])
            v = valid[:, idx]
            nv = v.sum().clamp(min=1.0)
            a = adv_t[:, idx]
            ratio = torch.exp(new_logp - old_logp[:, idx])
            pg_loss = -(torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.CLIP_EPS, 1 + cfg.CLIP_EPS) * a) * v).sum() / nv
            v_loss = 0.5 * (((value - ret_t[:, idx]) ** 2) * v).sum() / nv
            ent = (entropy * v).sum() / nv
            loss = pg_loss + cfg.VF_COEF * v_loss - cfg.ENT_COEF * ent

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.MAX_GRAD_NORM)
            opt.step()

            with torch.no_grad():
                stats["pg_loss"].append(pg_loss.item())
                stats["v_loss"].append(v_loss.item())
                stats["entropy"].append(ent.item())
                stats["approx_kl"].append((((ratio - 1) - torch.log(ratio)) * v).sum().item() / nv.item())
                stats["clip_frac"].append((((ratio - 1).abs() > cfg.CLIP_EPS).float() * v).sum().item() / nv.item())
    return {k: float(np.mean(vals)) for k, vals in stats.items()}


# ======================================================================
# main loop
# ======================================================================
recent = {k: deque(maxlen=cfg.WINRATE_WINDOW) for k in ["attacker_win", "planted", "defused", "length", "outs"]}
recent_bot = {team: deque(maxlen=cfg.WINRATE_WINDOW) for team in TEAMS}   # trained team's win rate v the bots
ep_outs = np.zeros(E)

steps_done, games_done, update = 0, 0, 0
next_record = 0
csv_path = os.path.join(run_dir, "metrics.csv")
csv_fields = ["update", "steps", "games", "attacker_win_rate", "plant_rate", "defuse_rate", "outs_per_round",
              "avg_round_length", "attackers_win_vs_bots", "defenders_win_vs_bots", "attacker_entropy", "defender_entropy",
              "attacker_v_loss", "defender_v_loss", "attacker_kl", "defender_kl", "steps_per_sec"]
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow(csv_fields)
bots_csv = os.path.join(run_dir, "bots.csv")
with open(bots_csv, "w", newline="") as f:
    csv.writer(f).writerow(["steps", "games", "trained_attackers_vs_bot_defenders", "bot_attackers_vs_trained_defenders"])

start_time = time.time()
print(f"training for {total_steps:,} steps ({E} games in parallel, {E * T:,} steps per update)")
print(f"bots play one team in {cfg.BOT_TEAM_PROB:.0%} of games; opponent pool: {cfg.POOL_PROB:.0%} of the rest\n")

try:
    while steps_done < total_steps:
        # pick which past snapshot each team uses when it plays as a past version during this rollout
        if cfg.OPPONENT_POOL:
            load_past_opponents()

        # ---------------- 1. collect experience ----------------
        for t in range(T):
            actions = {}
            for team in TEAMS:
                n = N_AGENTS[team]
                o_np = stack_team(obs, team)
                o_t = to_device(o_np)
                h_in = hidden[team]
                with torch.no_grad():
                    a, logp, value, hidden[team] = nets[team].act(o_t, h_in)
                act = a.cpu().numpy().reshape(E, n, ACTION_PARTS)

                # games where this team plays as a past version: use the past network's actions
                past_here = np.array([past_team[e] == team for e in range(E)])
                if past_here.any():
                    with torch.no_grad():
                        past_a, _, _, past_hidden[team] = past_nets[team].act(o_t, past_hidden[team])
                    act = np.where(past_here[:, None, None], past_a.cpu().numpy().reshape(E, n, ACTION_PARTS), act)

                b = buf[team]
                for key in OBS_KEYS:
                    b[key][t] = o_np[key].reshape((E, n) + OBS_SHAPES[key])
                if h_in is not None:
                    b["h"][t] = h_in.cpu().numpy().reshape(E, n, cfg.HIDDEN)
                b["action"][t] = act
                b["logp"][t] = logp.cpu().numpy().reshape(E, n)
                b["value"][t] = value.cpu().numpy().reshape(E, n)
                actions[team] = act

            # all games step at once in the workers (bots fill in their team); finished games come back already reset
            setup_before = game_t < cfg.SETUP_STEPS
            alive_before = {team: game_alive[team].copy() for team in TEAMS}
            new_obs, rewards, dones, infos = games.step(actions, bot_team)
            round_end[t] = np.array(dones, dtype=np.float32)
            # wipe the memory of every player whose round just ended (the next step is a new round)
            if round_end[t].any():
                for team in TEAMS:
                    keep_mem = torch.as_tensor(np.repeat(1.0 - round_end[t], N_AGENTS[team]), dtype=torch.float32,
                                               device=device).unsqueeze(1)
                    if hidden[team] is not None:
                        hidden[team] = hidden[team] * keep_mem
                    if past_hidden[team] is not None:
                        past_hidden[team] = past_hidden[team] * keep_mem

            for e in range(E):
                o, r, done, info = new_obs[e], rewards[e], dones[e], infos[e]
                for team in TEAMS:
                    # steps count for learning only for a team playing as its current self, while its player is in
                    # (and for attackers, not during the setup phase, when they are frozen)
                    learns = past_team[e] != team and bot_team[e] != team and team in cfg.TRAIN_TEAMS
                    frozen = team == "attackers" and setup_before[e]
                    buf[team]["valid"][t, e] = alive_before[team][e].astype(np.float32) * float(learns and not frozen)
                    taken_out = alive_before[team][e] & ~info["alive"][team]
                    buf[team]["reward"][t, e] = r[team]
                    buf[team]["done"][t, e] = (done | taken_out).astype(np.float32)
                ep_outs[e] += sum(info["outs"].values())

                if done:
                    games_done += 1
                    attacker_won = 1.0 if info["winner"] == "attackers" else 0.0
                    if past_team[e] is None and bot_team[e] is None:
                        # self-play stats only count games where both teams are their current selves
                        recent["attacker_win"].append(attacker_won)
                        recent["planted"].append(float(info["planted"]))
                        recent["defused"].append(float(info["defused"]))
                        recent["length"].append(info["t"])
                        recent["outs"].append(ep_outs[e])
                    elif bot_team[e] is not None:
                        trained = [t for t in TEAMS if t != bot_team[e]][0]
                        recent_bot[trained].append(attacker_won if trained == "attackers" else 1.0 - attacker_won)
                    ep_outs[e] = 0
                    past_team[e], bot_team[e] = pick_opponents()
                    game_t[e] = 0                            # the worker already started the next round
                    for team in TEAMS:
                        game_alive[team][e] = True
                else:
                    game_t[e] = info["t"]
                    for team in TEAMS:
                        game_alive[team][e] = info["alive"][team]
                obs[e] = o
            steps_done += E

        # ---------------- 2 + 3. advantages and PPO update, per team ----------------
        stats = {team: {} for team in TEAMS}
        for team in cfg.TRAIN_TEAMS:
            with torch.no_grad():
                _, _, last_value, _ = nets[team].act(to_device(stack_team(obs, team)), hidden[team])
            last_value = last_value.cpu().numpy().reshape(E, N_AGENTS[team])
            stats[team] = ppo_update(team, buf[team], last_value)
        update += 1

        # every snapshot_every updates, freeze a copy of each learning team into the pool.
        # When the pool is full, keep every second snapshot and take snapshots half as often from then on,
        # so the pool always covers the whole run evenly (old, weaker opponents included), not just the latest part.
        if cfg.OPPONENT_POOL and update % snapshot_every == 0:
            for team in cfg.TRAIN_TEAMS:
                pool[team].append(copy.deepcopy(nets[team].state_dict()))
            if len(pool[cfg.TRAIN_TEAMS[0]]) > cfg.POOL_SIZE:
                for team in cfg.TRAIN_TEAMS:
                    pool[team] = pool[team][::2]
                snapshot_every *= 2
                print(f"  -> opponent pool full: kept every second snapshot, now one every {snapshot_every} updates")

        # ---------------- logging ----------------
        sps = steps_done / (time.time() - start_time)
        if len(recent["attacker_win"]) > 0:
            mean = lambda d: float(np.mean(d)) if len(d) else np.nan
            row = {
                "update": update,
                "steps": steps_done,
                "games": games_done,
                "attacker_win_rate": mean(recent["attacker_win"]),
                "plant_rate": mean(recent["planted"]),
                "defuse_rate": mean(recent["defused"]),
                "outs_per_round": mean(recent["outs"]),
                "avg_round_length": mean(recent["length"]),
                "attackers_win_vs_bots": mean(recent_bot["attackers"]),
                "defenders_win_vs_bots": mean(recent_bot["defenders"]),
                "attacker_entropy": stats["attackers"].get("entropy", np.nan),
                "defender_entropy": stats["defenders"].get("entropy", np.nan),
                "attacker_v_loss": stats["attackers"].get("v_loss", np.nan),
                "defender_v_loss": stats["defenders"].get("v_loss", np.nan),
                "attacker_kl": stats["attackers"].get("approx_kl", np.nan),
                "defender_kl": stats["defenders"].get("approx_kl", np.nan),
                "steps_per_sec": sps,
            }
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([row[k] for k in csv_fields])
            print(f"update {update:4d} | steps {steps_done:>10,} | games {games_done:>6,} | "
                  f"attackers win {row['attacker_win_rate']:5.1%} (plant {row['plant_rate']:4.0%}, defuse {row['defuse_rate']:4.0%}) | "
                  f"outs {row['outs_per_round']:4.2f} | length {row['avg_round_length']:5.1f} | "
                  f"v bots: att {row['attackers_win_vs_bots']:4.0%} def {row['defenders_win_vs_bots']:4.0%} | "
                  f"entropy A {row['attacker_entropy']:.2f} D {row['defender_entropy']:.2f} | {sps:,.0f} steps/s")

        # ---------------- recording ----------------
        if games_done >= next_record:
            winner = record(games_done, steps_done)
            plot_curves()
            print(f"  -> recorded game {games_done:,} ({winner} won), saved GIF, checkpoint and curves")
            next_record += cfg.RECORD_EVERY

except KeyboardInterrupt:
    print("\nstopped by you (Ctrl+C), saving before exit")

games.close()
record(games_done, steps_done)
plot_curves()
minutes = (time.time() - start_time) / 60
print(f"\ndone: {steps_done:,} steps, {games_done:,} games in {minutes:.1f} min")
print(f"everything is in {run_dir}")
