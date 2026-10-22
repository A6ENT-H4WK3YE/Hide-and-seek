# Play rounds with random players or the scripted bots.
#  1. records one round as a GIF so you can watch it
#  2. plays many rounds without drawing, to get baseline win rates and speed
#
# Usage:
#   python random_demo.py                     bots v bots on the map from config
#   python random_demo.py random random       (attackers' policy, defenders' policy: "bot" or "random")

import os
import sys
import time
import numpy as np
import config as cfg
from env import GameEnv
from bots import Bots
from matches import play_match

a_pol = sys.argv[1] if len(sys.argv) > 1 else "bot"
d_pol = sys.argv[2] if len(sys.argv) > 2 else "bot"
n_rounds = 300

env = GameEnv(cfg.MAP_FILE)
bots = Bots(env)
os.makedirs(cfg.RUN_DIR, exist_ok=True)
name = os.path.splitext(os.path.basename(cfg.MAP_FILE))[0]
gif_path = os.path.join(cfg.RUN_DIR, f"demo_{a_pol}_attackers_v_{d_pol}_defenders_{name}.gif")
play_match(env, bots, {"attackers": a_pol, "defenders": d_pol}, None, 1, None, seed=cfg.FIXED_EVAL_SEED, gif_path=gif_path)
print(f"recorded one round -> {gif_path}")

t0 = time.time()
res = play_match(env, bots, {"attackers": a_pol, "defenders": d_pol}, None, n_rounds, None, seed=1)
elapsed = time.time() - t0
print(f"\n{n_rounds} rounds, {a_pol} attackers v {d_pol} defenders ({cfg.N_ATTACKERS}v{cfg.N_DEFENDERS}):")
print(f"  attackers win       {res['attacker_win']:.0%}")
print(f"  spike planted       {res['planted']:.0%}, defused {res['defused']:.0%}")
print(f"  players taken out   {res['outs_by_attackers']:.2f} by attackers, {res['outs_by_defenders']:.2f} by defenders per round")
print(f"  avg round length    {res['length']:.0f} steps")
print(f"  speed               {n_rounds * res['length'] / elapsed:,.0f} steps per second")
