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
from env import HideSeekEnv, load_map
from workers import ParallelGames
from model import TeamNet, get_device, OBS_KEYS, ACTION_PARTS, obs_to_tensors
from render import draw_frame, save_gif
from bots import Bots
from matches import play_match

assert not cfg.RANDOM_TEAM_SIZES, "random team sizes come later: set RANDOM_TEAM_SIZES = False"

TEAMS = ["seekers", "hiders"]
E, T = cfg.N_ENVS, cfg.ROLLOUT_STEPS
N_AGENTS = {"seekers": cfg.N_SEEKERS, "hiders": cfg.N_HIDERS}
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
    f.write(f"- game version: {cfg.GAME_VERSION} (see formulation.md)\n")
    f.write(f"- git commit: {git_commit()}\n")
    f.write(f"- device: {device}, {cfg.N_WORKERS} game workers\n")
    f.write(f"- map: {cfg.MAP_FILE}, teams: {cfg.N_SEEKERS} seekers v {cfg.N_HIDERS} hiders\n")
    f.write(f"- observation: rays ({cfg.N_RAYS}, 4) + enemies ({cfg.MAX_TEAM_SIZE}, 4) + heard ({cfg.MAX_TEAM_SIZE}, 4) + vec (7,) + minimap, "
            f"actions: move {cfg.N_MOVES} x turn {cfg.N_TURNS} {cfg.TURN_ANGLES} x keep/switch walk-run mode, "
            f"walk speed x{cfg.WALK_SPEED_FACTOR}, hearing range {cfg.HEARING_FRACTION} x map size = {cfg.HEARING_FRACTION * max(load_map(cfg.MAP_FILE)[1].shape):.2f} (through walls)\n")
    f.write(f"- rewards (seekers; hiders get minus): seen {cfg.REWARD_SEEN:+} x closeness, unseen {cfg.REWARD_UNSEEN:+}, "
            f"per catch {cfg.REWARD_CATCH:+}, win {cfg.REWARD_WIN:+} plus up to {cfg.REWARD_TIME_BONUS:+} for time left, "
            f"loss {cfg.REWARD_LOSS:+}\n")
    f.write(f"- learning: {', '.join(cfg.TRAIN_TEAMS)}"
            + (f" (the others are frozen, loaded from {cfg.FROZEN_RUN})" if len(cfg.TRAIN_TEAMS) < 2 else " (self-play)")
            + f"; hider noise cost {cfg.HIDER_NOISE_COST}, switch starts at keep {cfg.SWITCH_KEEP_PROB:.0%}\n")
    f.write(f"- total steps: {total_steps:,}\n\n")
    f.write("## All settings\n\n")
    for k, v in settings.items():
        f.write(f"- `{k}` = {v!r}\n")
print(f"run: {run_dir}   device: {device}")


# ======================================================================
# environments, networks, optimisers
# ======================================================================
# the E training games run in parallel worker processes (see workers.py), one per CPU core
games = ParallelGames(cfg.MAP_FILE, [cfg.SEED + i for i in range(E)], cfg.N_SEEKERS, cfg.N_HIDERS, cfg.N_WORKERS)
obs = games.reset()
# what this process needs to know about each game before stepping it (the games themselves live in the workers)
game_t = np.zeros(E, dtype=int)                                  # step count
game_alive = np.ones((E, cfg.N_HIDERS), dtype=bool)             # which hiders are still in
eval_env = HideSeekEnv(cfg.MAP_FILE)

nets = {team: TeamNet().to(device) for team in TEAMS}
opts = {team: torch.optim.Adam(nets[team].parameters(), lr=cfg.LEARNING_RATE, eps=1e-5) for team in cfg.TRAIN_TEAMS}

# ---------- opponent pool ----------
# pool[team] holds frozen snapshots of that team's network from earlier in training.
# past_nets[team] is the snapshot that team plays with, in games where it is "frozen".
pool = {team: [copy.deepcopy(nets[team].state_dict())] for team in cfg.TRAIN_TEAMS}
snapshot_every = cfg.SNAPSHOT_EVERY     # doubles each time the pool fills up (see the main loop)
past_nets = {team: TeamNet().to(device) for team in TEAMS}
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
    print(f"frozen {', '.join(FROZEN_TEAMS)} from {cfg.FROZEN_RUN}: final network, plus {len(picks)} checkpoints "
          f"(games {loaded[0]['games']:,} to {loaded[-1]['games']:,}) as their pool")
# in each game, at most one team plays as a past version: either team in self-play, only a frozen team otherwise
POOL_TEAMS = FROZEN_TEAMS if FROZEN_TEAMS else TEAMS


def pick_frozen_team():
    # for a new game: None (both teams current, normal self-play) or the team that plays as a past version
    if cfg.OPPONENT_POOL and pool_rng.random() < cfg.POOL_PROB:
        return str(pool_rng.choice(POOL_TEAMS))
    return None


def load_past_opponents():
    for team in TEAMS:
        past_nets[team].load_state_dict(pool[team][pool_rng.integers(len(pool[team]))])


frozen_team = [pick_frozen_team() for _ in range(E)]

# ---------- benchmark bots ----------
bot_env = HideSeekEnv(cfg.MAP_FILE)
bots = Bots(bot_env)

# reference numbers for THIS map and team size, measured once so the curves have the right yardsticks
REF = {}
if cfg.BOT_EVAL_GAMES > 0:
    print(f"measuring bot reference numbers for {cfg.N_SEEKERS}v{cfg.N_HIDERS} on {cfg.MAP_FILE}...")
    for s_pol, h_pol, key in [("random", "random", "random_vs_random"),
                              ("bot", "random", "chaser_vs_random"),
                              ("bot", "bot", "chaser_vs_flee")]:
        REF[key] = play_match(bot_env, bots, {"seekers": s_pol, "hiders": h_pol}, None, 300, device, seed=50_000)["seeker_win"]
    print(f"  random seekers vs random hiders: seekers win {REF['random_vs_random']:.0%}")
    print(f"  chaser bots vs random hiders:    seekers win {REF['chaser_vs_random']:.0%}")
    print(f"  chaser bots vs fleeing bots:     seekers win {REF['chaser_vs_flee']:.0%}")
    with open(os.path.join(run_dir, "run_card.md"), "a") as f:
        f.write("\n## Bot reference numbers (seeker win rate)\n\n")
        for k, val in REF.items():
            f.write(f"- {k}: {val:.1%}\n")


def stack_team(obs_all, team):
    # every agent of one team across all parallel games -> dict of arrays of shape (E * n, ...)
    flat = [o[team][k] for o in obs_all for k in range(len(o[team]))]
    return {key: np.stack([x[key] for x in flat]) for key in OBS_KEYS}


def to_device(arrays):
    return {key: torch.as_tensor(x, device=device) for key, x in arrays.items()}


# the shape of each part of one agent's observation, e.g. {"rays": (31, 4), ...}
OBS_SHAPES = {key: x.shape for key, x in obs[0]["seekers"][0].items()}


# storage for one rollout, per team: index [step, game, agent]
def new_buffer(n):
    return {
        **{key: np.zeros((T, E, n) + OBS_SHAPES[key], dtype=np.float32) for key in OBS_KEYS},
        "action": np.zeros((T, E, n, ACTION_PARTS), dtype=np.int64),     # (move, turn, speed) per agent
        "logp": np.zeros((T, E, n), dtype=np.float32),
        "value": np.zeros((T, E, n), dtype=np.float32),
        "reward": np.zeros((T, E, n), dtype=np.float32),
        "done": np.zeros((T, E, n), dtype=np.float32),
        # valid = this step counts for learning. Not valid: seekers during the hiding phase
        # (their actions do nothing), hiders after they've been caught, and any team playing
        # as a frozen past version (that data came from an old network, not the one learning).
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
            with torch.no_grad():
                a, _, _ = nets[team].act(to_device(stack_team([o], team)))
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

    # benchmark against the fixed bots: unlike self-play, these opponents never change,
    # so these numbers show real progress
    if cfg.BOT_EVAL_GAMES > 0:
        hiders_test = play_match(bot_env, bots, {"seekers": "bot", "hiders": "trained"}, nets, cfg.BOT_EVAL_GAMES, device)
        seekers_test = play_match(bot_env, bots, {"seekers": "trained", "hiders": "random"}, nets, cfg.BOT_EVAL_GAMES, device)
        with open(bots_csv, "a", newline="") as f:
            csv.writer(f).writerow([steps_done, games_done, hiders_test["seeker_win"], seekers_test["seeker_win"]])
        print(f"  -> vs bots: chaser bot beats trained hiders {hiders_test['seeker_win']:.0%} "
              f"(random hider {REF['chaser_vs_random']:.0%}, fleeing bot {REF['chaser_vs_flee']:.0%}) | "
              f"trained seekers beat random hiders {seekers_test['seeker_win']:.0%} "
              f"(random seekers {REF['random_vs_random']:.0%}, chaser bots {REF['chaser_vs_random']:.0%})")
    return info["winner"]


def plot_curves():
    path = os.path.join(run_dir, "metrics.csv")
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size < 2:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    x = data["steps"]
    axes[0, 0].plot(x, data["seeker_win_rate"], color="#c0392b")
    if REF:
        axes[0, 0].axhline(REF["random_vs_random"], color="grey", linestyle="--", linewidth=1, label="random vs random")
    axes[0, 0].set_title("self-play: seeker win rate (recent games)")
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

    # benchmark panel
    ax = axes[0, 2]
    if os.path.exists(bots_csv):
        bd = np.atleast_1d(np.genfromtxt(bots_csv, delimiter=",", names=True))
        if bd.size >= 1:
            ax.plot(bd["steps"], bd["chaser_vs_trained_hiders"], "o-", color="#2563b0",
                    label="chaser bot beats trained hiders (lower = better hiders)")
            ax.plot(bd["steps"], bd["trained_seekers_vs_random_hider"], "o-", color="#c0392b",
                    label="trained seekers beat random hider (higher = better seekers)")
    for key, label in [("chaser_vs_random", "chaser bots vs random hiders"),
                       ("random_vs_random", "random vs random"),
                       ("chaser_vs_flee", "chaser bots vs fleeing bots")]:
        if key in REF:
            ax.axhline(REF[key], color="grey", linestyle=":", linewidth=1)
            ax.text(0, REF[key] + 0.015, f"{label} ({REF[key]:.0%})", fontsize=7, color="grey")
    ax.set_title("vs fixed benchmark bots")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper right")
    axes[1, 2].axis("off")
    axes[1, 2].text(0, 0.9, "How to read this", fontsize=10, fontweight="bold")
    axes[1, 2].text(0, 0.05,
                    "Self-play win rate can go up and down forever,\n"
                    "because both teams keep changing.\n\n"
                    "The bot panel uses opponents that never change.\n"
                    "Blue: hiders that learn to run away push it down\n"
                    "towards the 'chaser vs fleeing bots' line or below.\n"
                    "Red: seekers that learn to search and chase push it\n"
                    "up towards the 'chaser vs random hiders' line.\n"
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
bots_csv = os.path.join(run_dir, "bots.csv")
with open(bots_csv, "w", newline="") as f:
    csv.writer(f).writerow(["steps", "games", "chaser_vs_trained_hiders", "trained_seekers_vs_random_hider"])

start_time = time.time()
print(f"training for {total_steps:,} steps ({E} games in parallel, {E * T:,} steps per update)")
if cfg.OPPONENT_POOL:
    print(f"opponent pool on: {cfg.POOL_PROB:.0%} of games use a past version of "
          + ("one team\n" if not FROZEN_TEAMS else f"the frozen {', '.join(FROZEN_TEAMS)}\n"))
else:
    print("opponent pool off: plain self-play\n")

try:
    while steps_done < total_steps:
        # pick which past snapshot each team uses when it plays "frozen" during this rollout
        if cfg.OPPONENT_POOL:
            load_past_opponents()

        # ---------------- 1. collect experience ----------------
        for t in range(T):
            actions = {}
            for team in TEAMS:
                n = N_AGENTS[team]
                o_np = stack_team(obs, team)
                o_t = to_device(o_np)
                with torch.no_grad():
                    a, logp, value = nets[team].act(o_t)
                act = a.cpu().numpy().reshape(E, n, ACTION_PARTS)

                # games where this team plays as a frozen past version: use the past network's actions
                frozen_here = np.array([frozen_team[e] == team for e in range(E)])
                if frozen_here.any():
                    with torch.no_grad():
                        past_a, _, _ = past_nets[team].act(o_t)
                    act = np.where(frozen_here[:, None, None], past_a.cpu().numpy().reshape(E, n, ACTION_PARTS), act)

                b = buf[team]
                for key in OBS_KEYS:
                    b[key][t] = o_np[key].reshape((E, n) + OBS_SHAPES[key])
                b["action"][t] = act
                b["logp"][t] = logp.cpu().numpy().reshape(E, n)
                b["value"][t] = value.cpu().numpy().reshape(E, n)
                actions[team] = act

            # all games step at once in the workers; finished games come back already reset
            hunting_before = game_t >= cfg.HIDING_PHASE
            alive_before_all = game_alive.copy()
            new_obs, rewards, dones, infos = games.step(actions["seekers"], actions["hiders"])

            for e in range(E):
                hunting, alive_before = hunting_before[e], alive_before_all[e]
                o, r, done, info = new_obs[e], rewards[e], dones[e], infos[e]
                # frozen past versions (and teams that do not learn at all) don't learn, so their steps never count
                buf["seekers"]["valid"][t, e] = float(hunting and frozen_team[e] != "seekers" and "seekers" in cfg.TRAIN_TEAMS)
                buf["hiders"]["valid"][t, e] = alive_before.astype(np.float32) * float(
                    frozen_team[e] != "hiders" and "hiders" in cfg.TRAIN_TEAMS)

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
                    # self-play stats only count games where both teams are their current selves
                    if frozen_team[e] is None:
                        recent_winner.append(1.0 if info["winner"] == "seekers" else 0.0)
                        recent_len.append(info["t"])
                        recent_spotted.append(ep_spotted[e] / max(ep_hunting[e], 1))
                        recent_hider_return.append(ep_hider_return[e])
                    ep_hider_return[e], ep_spotted[e], ep_hunting[e] = 0, 0, 0
                    frozen_team[e] = pick_frozen_team()
                    game_t[e], game_alive[e] = 0, True       # the worker already started the next round
                else:
                    game_t[e], game_alive[e] = info["t"], info["hider_alive"]
                obs[e] = o
            steps_done += E

        # ---------------- 2 + 3. advantages and PPO update, per team ----------------
        stats = {team: {} for team in TEAMS}
        for team in cfg.TRAIN_TEAMS:
            with torch.no_grad():
                _, _, last_value = nets[team].act(to_device(stack_team(obs, team)))
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

games.close()
record(games_done, steps_done)
plot_curves()
minutes = (time.time() - start_time) / 60
print(f"\ndone: {steps_done:,} steps, {games_done:,} games in {minutes:.1f} min")
print(f"everything is in {run_dir}")
