# PPO self-play training: seekers and hiders learn at the same time, against each other.
#
# Usage:
#   python train.py                    full run (TOTAL_STEPS from config)
#   python train.py smoke 20000        quick check that everything runs (name, steps)
#
# Output goes to runs/<run name>/:
#   run_card.md      every setting used, the game version and the git commit
#   metrics.csv      one row per update (win rate, rewards, losses, ...)
#   curves.png       plots of metrics.csv, refreshed every recording
#   gifs/            one recorded game every RECORD_EVERY games
#   checkpoints/     network weights saved with each GIF, plus latest.pt
#
# The loop, in short:
#   1. play N_ENVS games in parallel for ROLLOUT_STEPS steps, storing what happened
#   2. work out for every step how much better or worse things went than expected (advantages, GAE)
#   3. nudge each team's network towards actions that went better than expected (PPO update)
#   4. repeat

import os
import sys
import csv
import time
import datetime
import subprocess
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from env import HideSeekEnv, N_CHANNELS
from model import TeamNet, get_device
from render import draw_frame, save_gif

assert not cfg.RANDOM_TEAM_SIZES, "random team sizes come later: set RANDOM_TEAM_SIZES = False"

TEAMS = ["seekers", "hiders"]
E, T = cfg.N_ENVS, cfg.ROLLOUT_STEPS
N_AGENTS = {"seekers": cfg.N_SEEKERS, "hiders": cfg.N_HIDERS}
W = cfg.OBS_WINDOW
total_steps = int(sys.argv[2]) if len(sys.argv) > 2 else cfg.TOTAL_STEPS

np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
device = get_device()


# ======================================================================
# run folder and run card
# ======================================================================
map_name = os.path.splitext(os.path.basename(cfg.MAP_FILE))[0]
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
default_name = f"{cfg.GAME_VERSION}_{map_name}_{cfg.N_SEEKERS}v{cfg.N_HIDERS}_{stamp}"
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
    f.write(f"- game version: {cfg.GAME_VERSION} (see docs/formulation.md)\n")
    f.write(f"- git commit: {git_commit()}\n")
    f.write(f"- device: {device}\n")
    f.write(f"- map: {cfg.MAP_FILE}, teams: {cfg.N_SEEKERS} seekers v {cfg.N_HIDERS} hiders\n")
    f.write(f"- observation: grid ({N_CHANNELS}, {W}, {W}) + vec (4,), actions: {cfg.N_ACTIONS}\n")
    f.write(f"- rewards (hiders; seekers get minus): hidden {cfg.REWARD_HIDDEN:+}, seen {cfg.REWARD_SEEN:+}, "
            f"per catch {-cfg.REWARD_CATCH:+}, win/loss +-{cfg.REWARD_WIN}\n")
    f.write(f"- total steps: {total_steps:,}\n\n")
    f.write("## All settings\n\n")
    for k, v in settings.items():
        f.write(f"- `{k}` = {v!r}\n")
print(f"run: {run_dir}   device: {device}")


# ======================================================================
# environments, networks, optimisers
# ======================================================================
envs = [HideSeekEnv(cfg.MAP_FILE) for _ in range(E)]
obs = [env.reset(seed=cfg.SEED + i, n_seekers=cfg.N_SEEKERS, n_hiders=cfg.N_HIDERS) for i, env in enumerate(envs)]
eval_env = HideSeekEnv(cfg.MAP_FILE)

nets = {team: TeamNet().to(device) for team in TEAMS}
opts = {team: torch.optim.Adam(nets[team].parameters(), lr=cfg.LEARNING_RATE, eps=1e-5) for team in TEAMS}


def stack_team(obs_all, team):
    # every agent of one team across all parallel games -> arrays of shape (E * n, ...)
    flat = [o[team][k] for o in obs_all for k in range(len(o[team]))]
    return np.stack([x["grid"] for x in flat]), np.stack([x["vec"] for x in flat])


# storage for one rollout, per team: index [step, game, agent]
def new_buffer(n):
    return {
        "grid": np.zeros((T, E, n, N_CHANNELS, W, W), dtype=np.float32),
        "vec": np.zeros((T, E, n, 4), dtype=np.float32),
        "action": np.zeros((T, E, n), dtype=np.int64),
        "logp": np.zeros((T, E, n), dtype=np.float32),
        "value": np.zeros((T, E, n), dtype=np.float32),
        "reward": np.zeros((T, E, n), dtype=np.float32),
        "done": np.zeros((T, E, n), dtype=np.float32),
        # valid = this step counts for learning. Not valid: seekers during the hiding phase
        # (their actions do nothing) and hiders after they've been caught.
        "valid": np.zeros((T, E, n), dtype=np.float32),
    }


buf = {team: new_buffer(N_AGENTS[team]) for team in TEAMS}


# ======================================================================
# recording: one game as a GIF + checkpoint, and the curves plot
# ======================================================================
def record(games_done, steps_done):
    o = eval_env.reset(seed=cfg.FIXED_EVAL_SEED, n_seekers=cfg.N_SEEKERS, n_hiders=cfg.N_HIDERS)
    frames = [draw_frame(eval_env)]
    done = False
    while not done:
        acts = {}
        for team in TEAMS:
            g, v = stack_team([o], team)
            with torch.no_grad():
                a, _, _ = nets[team].act(torch.as_tensor(g, device=device), torch.as_tensor(v, device=device))
            acts[team] = a.cpu().numpy()
        o, _, done, info = eval_env.step(acts["seekers"], acts["hiders"])
        frames.append(draw_frame(eval_env))
    save_gif(frames, os.path.join(run_dir, "gifs", f"game_{games_done:07d}.gif"))

    ckpt = {
        "seekers": nets["seekers"].state_dict(),
        "hiders": nets["hiders"].state_dict(),
        "games": games_done,
        "steps": steps_done,
        "game_version": cfg.GAME_VERSION,
        "settings": settings,
    }
    torch.save(ckpt, os.path.join(run_dir, "checkpoints", f"ckpt_{games_done:07d}.pt"))
    torch.save(ckpt, os.path.join(run_dir, "checkpoints", "latest.pt"))
    return info["winner"]


def plot_curves():
    path = os.path.join(run_dir, "metrics.csv")
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size < 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = data["steps"]
    axes[0, 0].plot(x, data["seeker_win_rate"], color="#c0392b")
    axes[0, 0].axhline(0.29, color="grey", linestyle="--", linewidth=1, label="random agents (1v1)")
    axes[0, 0].set_title("seeker win rate (recent games)")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()
    axes[0, 1].plot(x, data["hider_return"], color="#2563b0")
    axes[0, 1].set_title("hider return per game (seekers get minus this)")
    axes[1, 0].plot(x, data["spotted_frac"], color="#7a5c00")
    axes[1, 0].set_title("fraction of hunting steps a hider is spotted")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 1].plot(x, data["seeker_entropy"], color="#c0392b", label="seekers")
    axes[1, 1].plot(x, data["hider_entropy"], color="#2563b0", label="hiders")
    axes[1, 1].set_title("policy entropy (high = random, low = decisive)")
    axes[1, 1].legend()
    for ax in axes.flat:
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
    # Worked backwards from the last step. done cuts the chain (no future after the game or a catch).
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

    # --- keep only the steps that count, as flat arrays ---
    keep = b["valid"].reshape(-1) > 0
    n = int(keep.sum())
    if n < 2:
        return {}
    to_t = lambda x, dtype=torch.float32: torch.as_tensor(x.reshape(T * E * N_AGENTS[team], *x.shape[3:])[keep], dtype=dtype, device=device)
    g, v = to_t(b["grid"]), to_t(b["vec"])
    actions = to_t(b["action"], torch.int64)
    old_logp, adv_t, ret_t = to_t(b["logp"]), to_t(adv), to_t(returns)

    # --- PPO: several passes over the data in minibatches ---
    net, opt = nets[team], opts[team]
    mb_size = max(n // cfg.MINIBATCHES, 1)
    stats = {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    for epoch in range(cfg.PPO_EPOCHS):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, mb_size):
            idx = perm[start:start + mb_size]
            logits, value = net(g[idx], v[idx])
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(actions[idx])
            entropy = dist.entropy().mean()

            # normalise advantages inside the minibatch (keeps step sizes sensible)
            a = adv_t[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)

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


# ======================================================================
# main loop
# ======================================================================
recent_winner = deque(maxlen=cfg.WINRATE_WINDOW)     # 1 if seekers won
recent_len = deque(maxlen=cfg.WINRATE_WINDOW)
recent_spotted = deque(maxlen=cfg.WINRATE_WINDOW)
recent_hider_return = deque(maxlen=cfg.WINRATE_WINDOW)
ep_hider_return = np.zeros(E)
ep_spotted = np.zeros(E)
ep_hunting = np.zeros(E)

steps_done, games_done, update = 0, 0, 0
next_record = 0
csv_path = os.path.join(run_dir, "metrics.csv")
csv_fields = ["update", "steps", "games", "seeker_win_rate", "avg_game_length", "spotted_frac", "hider_return",
              "seeker_entropy", "hider_entropy", "seeker_v_loss", "hider_v_loss", "seeker_kl", "hider_kl", "steps_per_sec"]
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow(csv_fields)

start_time = time.time()
print(f"training for {total_steps:,} steps ({E} games in parallel, {E * T:,} steps per update)\n")

try:
    while steps_done < total_steps:
        # ---------------- 1. collect experience ----------------
        for t in range(T):
            actions = {}
            for team in TEAMS:
                n = N_AGENTS[team]
                g, v = stack_team(obs, team)
                with torch.no_grad():
                    a, logp, value = nets[team].act(torch.as_tensor(g, device=device), torch.as_tensor(v, device=device))
                b = buf[team]
                b["grid"][t] = g.reshape(E, n, N_CHANNELS, W, W)
                b["vec"][t] = v.reshape(E, n, 4)
                b["action"][t] = a.cpu().numpy().reshape(E, n)
                b["logp"][t] = logp.cpu().numpy().reshape(E, n)
                b["value"][t] = value.cpu().numpy().reshape(E, n)
                actions[team] = b["action"][t]

            for e, env in enumerate(envs):
                hunting = env.t >= cfg.HIDING_PHASE
                alive_before = env.hider_alive.copy()
                buf["seekers"]["valid"][t, e] = float(hunting)
                buf["hiders"]["valid"][t, e] = alive_before.astype(np.float32)

                o, r, done, info = env.step(actions["seekers"][e], actions["hiders"][e])

                newly_caught = alive_before & ~info["hider_alive"]
                buf["seekers"]["reward"][t, e] = r["seekers"]
                buf["hiders"]["reward"][t, e] = r["hiders"]
                buf["seekers"]["done"][t, e] = float(done)
                buf["hiders"]["done"][t, e] = (done | newly_caught).astype(np.float32)

                ep_hider_return[e] += r["hiders"][0]
                if hunting:
                    ep_hunting[e] += 1
                    ep_spotted[e] += info["any_seen"]

                if done:
                    games_done += 1
                    recent_winner.append(1.0 if info["winner"] == "seekers" else 0.0)
                    recent_len.append(info["t"])
                    recent_spotted.append(ep_spotted[e] / max(ep_hunting[e], 1))
                    recent_hider_return.append(ep_hider_return[e])
                    ep_hider_return[e], ep_spotted[e], ep_hunting[e] = 0, 0, 0
                    o = env.reset(n_seekers=cfg.N_SEEKERS, n_hiders=cfg.N_HIDERS)
                obs[e] = o
            steps_done += E

        # ---------------- 2 + 3. advantages and PPO update, per team ----------------
        stats = {}
        for team in TEAMS:
            g, v = stack_team(obs, team)
            with torch.no_grad():
                _, _, last_value = nets[team].act(torch.as_tensor(g, device=device), torch.as_tensor(v, device=device))
            last_value = last_value.cpu().numpy().reshape(E, N_AGENTS[team])
            stats[team] = ppo_update(team, buf[team], last_value)
        update += 1

        # ---------------- logging ----------------
        sps = steps_done / (time.time() - start_time)
        if len(recent_winner) > 0:
            row = {
                "update": update,
                "steps": steps_done,
                "games": games_done,
                "seeker_win_rate": np.mean(recent_winner),
                "avg_game_length": np.mean(recent_len),
                "spotted_frac": np.mean(recent_spotted),
                "hider_return": np.mean(recent_hider_return),
                "seeker_entropy": stats["seekers"].get("entropy", np.nan),
                "hider_entropy": stats["hiders"].get("entropy", np.nan),
                "seeker_v_loss": stats["seekers"].get("v_loss", np.nan),
                "hider_v_loss": stats["hiders"].get("v_loss", np.nan),
                "seeker_kl": stats["seekers"].get("approx_kl", np.nan),
                "hider_kl": stats["hiders"].get("approx_kl", np.nan),
                "steps_per_sec": sps,
            }
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([row[k] for k in csv_fields])
            print(f"update {update:4d} | steps {steps_done:>9,} | games {games_done:>6,} | "
                  f"seeker win {row['seeker_win_rate']:5.1%} | length {row['avg_game_length']:5.1f} | "
                  f"spotted {row['spotted_frac']:5.1%} | entropy S {row['seeker_entropy']:.2f} H {row['hider_entropy']:.2f} | "
                  f"{sps:,.0f} steps/s")

        # ---------------- recording ----------------
        if games_done >= next_record:
            winner = record(games_done, steps_done)
            plot_curves()
            print(f"  -> recorded game {games_done:,} ({winner} won), saved GIF, checkpoint and curves")
            next_record += cfg.RECORD_EVERY

except KeyboardInterrupt:
    print("\nstopped by you (Ctrl+C), saving before exit")

record(games_done, steps_done)
plot_curves()
minutes = (time.time() - start_time) / 60
print(f"\ndone: {steps_done:,} steps, {games_done:,} games in {minutes:.1f} min")
print(f"everything is in {run_dir}")
