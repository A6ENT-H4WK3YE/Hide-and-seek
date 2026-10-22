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
from env import GameEnv, TEAMS
from model import load_net, get_device
from bots import Bots
from matches import play_match

ckpt_path = sys.argv[1]
n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 300
device = get_device()

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
if ckpt.get("game_version") != cfg.GAME_VERSION:
    print(f"WARNING: checkpoint was trained on game {ckpt.get('game_version')}, config says {cfg.GAME_VERSION}")
nets = {}
for team in TEAMS:
    nets[team] = load_net(ckpt[team], device)     # with memory if the saved network has one
    nets[team].eval()
print(f"loaded {ckpt_path} (trained for {ckpt['games']:,} games, {ckpt['steps']:,} steps)\n")

env = GameEnv(cfg.MAP_FILE)
bots = Bots(env)
out_dir = os.path.dirname(os.path.dirname(ckpt_path))   # the run folder

matchups = [
    # (attackers, defenders, what it tells you)
    ("random", "random", "reference: nobody trained"),
    ("bot", "bot", "reference: bots against bots"),
    ("bot", "random", "reference: bot attackers against a careless defence"),
    ("random", "bot", "reference: careless attackers against bot defenders"),
    ("trained", "random", "ATTACKERS: higher = better"),
    ("trained", "bot", "ATTACKERS: higher = better (compare with bots v bots)"),
    ("random", "trained", "DEFENDERS: lower = better"),
    ("bot", "trained", "DEFENDERS: lower = better (compare with bots v bots)"),
    ("trained", "trained", "self-play matchup"),
]
print(f"{n_games} games per matchup, {cfg.N_ATTACKERS}v{cfg.N_DEFENDERS} on {cfg.MAP_FILE}\n")
print(f"{'attackers':>9} v {'defenders':<9} | att win | planted | defused | outs by A | outs by D | length | what it tells you")
print("-" * 120)
for a_pol, d_pol, meaning in matchups:
    gif = os.path.join(out_dir, f"eval_{a_pol}_attackers_v_{d_pol}_defenders.gif")
    res = play_match(env, bots, {"attackers": a_pol, "defenders": d_pol}, nets, n_games, device, gif_path=gif)
    print(f"{a_pol:>9} v {d_pol:<9} | {res['attacker_win']:7.1%} | {res['planted']:7.1%} | {res['defused']:7.1%} | "
          f"{res['outs_by_attackers']:9.2f} | {res['outs_by_defenders']:9.2f} | {res['length']:6.1f} | {meaning}")
print(f"\nGIFs of the first game of each matchup saved in {out_dir}")
