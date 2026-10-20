# Test a trained checkpoint by playing it against random opponents.
#
# Why: in self-play both teams improve together, so the win rate during training can sit near
# the same value even while both get much better. Playing each trained team against a RANDOM
# opponent shows whether it actually learned something.
#
# Usage:
#   python evaluate.py runs/<run name>/checkpoints/latest.pt
#   python evaluate.py runs/<run name>/checkpoints/latest.pt 1000      (number of games per matchup)

import os
import sys
import numpy as np
import torch

import config as cfg
from env import HideSeekEnv
from model import TeamNet, get_device
from render import draw_frame, save_gif

ckpt_path = sys.argv[1]
n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 500
device = get_device()

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
if ckpt.get("game_version") != cfg.GAME_VERSION:
    print(f"WARNING: checkpoint was trained on game {ckpt.get('game_version')}, config says {cfg.GAME_VERSION}")
nets = {}
for team in ["seekers", "hiders"]:
    nets[team] = TeamNet().to(device)
    nets[team].load_state_dict(ckpt[team])
    nets[team].eval()
print(f"loaded {ckpt_path} (trained for {ckpt['games']:,} games, {ckpt['steps']:,} steps)\n")

env = HideSeekEnv(cfg.MAP_FILE)
rng = np.random.default_rng(123)


def choose(team, obs_team, policy):
    # policy is "trained" or "random"
    n = len(obs_team)
    if policy == "random":
        return rng.integers(0, cfg.N_ACTIONS, size=n)
    g = torch.as_tensor(np.stack([o["grid"] for o in obs_team]), device=device)
    v = torch.as_tensor(np.stack([o["vec"] for o in obs_team]), device=device)
    with torch.no_grad():
        a, _, _ = nets[team].act(g, v)
    return a.cpu().numpy()


def play(seeker_policy, hider_policy, games, gif_path=None):
    seeker_wins, lengths, spotted, hunting = 0, [], 0, 0
    for k in range(games):
        obs = env.reset(seed=10_000 + k, n_seekers=cfg.N_SEEKERS, n_hiders=cfg.N_HIDERS)
        frames = [draw_frame(env)] if (gif_path and k == 0) else None
        done = False
        while not done:
            s_act = choose("seekers", obs["seekers"], seeker_policy)
            h_act = choose("hiders", obs["hiders"], hider_policy)
            obs, _, done, info = env.step(s_act, h_act)
            if not info["hiding_phase"]:
                hunting += 1
                spotted += info["any_seen"]
            if frames is not None:
                frames.append(draw_frame(env))
        seeker_wins += info["winner"] == "seekers"
        lengths.append(info["t"])
        if frames is not None:
            save_gif(frames, gif_path)
    return seeker_wins / games, np.mean(lengths), spotted / max(hunting, 1)


out_dir = os.path.dirname(os.path.dirname(ckpt_path))   # the run folder
matchups = [
    ("random", "random", "baseline"),
    ("trained", "random", "did seekers learn?  (should be well ABOVE baseline)"),
    ("random", "trained", "did hiders learn?   (should be well BELOW baseline)"),
    ("trained", "trained", "self-play matchup"),
]
print(f"{n_games} games per matchup, {cfg.N_SEEKERS}v{cfg.N_HIDERS} on {cfg.MAP_FILE}\n")
print(f"{'seekers':>8} vs {'hiders':<8} | seeker win | length | spotted | meaning")
print("-" * 90)
for s_pol, h_pol, meaning in matchups:
    gif = os.path.join(out_dir, f"eval_{s_pol}_seekers_vs_{h_pol}_hiders.gif")
    win, length, spot = play(s_pol, h_pol, n_games, gif_path=gif)
    print(f"{s_pol:>8} vs {h_pol:<8} | {win:9.1%} | {length:6.1f} | {spot:7.1%} | {meaning}")
print(f"\nGIFs of the first game of each matchup saved in {out_dir}")
