# Test a trained checkpoint against random players and the scripted bots (see bots.py).
#
# Why: in self-play both teams change together, so the training win rate can go up and down
# forever. Fixed opponents (random players, bots) show whether each team actually got better.
#
# Usage:
#   python evaluate.py runs/<run name>/checkpoints/latest.pt
#   python evaluate.py runs/<run name>/checkpoints/latest.pt 1000      (games per matchup)

import os
import sys
import torch

import config as cfg
from env import HideSeekEnv
from model import load_net, get_device
from bots import Bots
from matches import play_match

ckpt_path = sys.argv[1]
n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 500
device = get_device()

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
if ckpt.get("game_version") != cfg.GAME_VERSION:
    print(f"WARNING: checkpoint was trained on game {ckpt.get('game_version')}, config says {cfg.GAME_VERSION}")
nets = {}
for team in ["seekers", "hiders"]:
    nets[team] = load_net(ckpt[team], device)     # with memory if the saved network has one
    nets[team].eval()
print(f"loaded {ckpt_path} (trained for {ckpt['games']:,} games, {ckpt['steps']:,} steps)\n")

env = HideSeekEnv(cfg.MAP_FILE)
bots = Bots(env)
out_dir = os.path.dirname(os.path.dirname(ckpt_path))   # the run folder

matchups = [
    # (seekers, hiders, what it tells you)
    ("random", "random", "reference: nobody trained"),
    ("bot", "random", "reference: what a good chaser does to a careless hider"),
    ("bot", "bot", "reference: a fleeing hider escapes the chaser"),
    ("trained", "random", "SEEKERS: higher = better (compare with the chaser bot line)"),
    ("bot", "trained", "HIDERS: lower = better (compare with the two bot lines above)"),
    ("trained", "bot", "SEEKERS vs the fleeing bot: higher = better (compare with bot vs bot)"),
    ("bot", "escape", "reference: the chaser bot against the cheating escape bot (knows where seekers are)"),
    ("trained", "escape", "SEEKERS stress test: vs the cheating escape bot (compare with the line above)"),
    ("trained", "trained", "self-play matchup"),
]
print(f"{n_games} games per matchup, {cfg.N_SEEKERS}v{cfg.N_HIDERS} on {cfg.MAP_FILE}\n")
print(f"{'seekers':>8} vs {'hiders':<8} | seeker win | length | spotted | what it tells you")
print("-" * 100)
for s_pol, h_pol, meaning in matchups:
    gif = os.path.join(out_dir, f"eval_{s_pol}_seekers_vs_{h_pol}_hiders.gif")
    res = play_match(env, bots, {"seekers": s_pol, "hiders": h_pol}, nets, n_games, device, gif_path=gif)
    print(f"{s_pol:>8} vs {h_pol:<8} | {res['seeker_win']:9.1%} | {res['length']:6.1f} | {res['spotted']:7.1%} | {meaning}")
print(f"\nGIFs of the first game of each matchup saved in {out_dir}")
