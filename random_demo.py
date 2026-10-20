# Play rounds with completely random agents.
#  1. records one round as a GIF so you can watch it
#  2. plays many rounds without drawing, to get baseline win rates and speed
#
# Usage:
#   python random_demo.py                          (uses MAP_FILE from config)
#   python random_demo.py maps/three_sites.txt 3 3 (map, seekers, hiders)

import os
import sys
import time
import numpy as np
import config as cfg
from env import HideSeekEnv
from render import draw_frame, save_gif

map_file = sys.argv[1] if len(sys.argv) > 1 else cfg.MAP_FILE
n_seekers = int(sys.argv[2]) if len(sys.argv) > 2 else cfg.N_SEEKERS
n_hiders = int(sys.argv[3]) if len(sys.argv) > 3 else cfg.N_HIDERS
n_rounds = 300

t0 = time.time()
env = HideSeekEnv(map_file)
print(f"map {map_file} ({env.W} x {env.H}), line of sight table built in {time.time() - t0:.1f}s")
rng = np.random.default_rng(0)


def random_actions(n):
    return rng.integers(0, cfg.N_ACTIONS, size=n)


# ---------- 1. one recorded round ----------
os.makedirs(cfg.RUN_DIR, exist_ok=True)
env.reset(seed=cfg.FIXED_EVAL_SEED, n_seekers=n_seekers, n_hiders=n_hiders)
frames = [draw_frame(env)]
done = False
while not done:
    _, _, done, info = env.step(random_actions(n_seekers), random_actions(n_hiders))
    frames.append(draw_frame(env))
name = os.path.splitext(os.path.basename(map_file))[0]
gif_path = os.path.join(cfg.RUN_DIR, f"random_{name}_{n_seekers}v{n_hiders}.gif")
save_gif(frames, gif_path)
print(f"recorded round: {info['winner']} win after {info['t']} steps -> {gif_path}")

# ---------- 2. many rounds, no drawing ----------
seeker_wins, total_steps, seen_steps = 0, 0, 0
t0 = time.time()
for k in range(n_rounds):
    env.reset(n_seekers=n_seekers, n_hiders=n_hiders)
    done = False
    while not done:
        _, _, done, info = env.step(random_actions(n_seekers), random_actions(n_hiders))
        total_steps += 1
        seen_steps += info["any_seen"]
    seeker_wins += info["winner"] == "seekers"
elapsed = time.time() - t0

print(f"\n{n_rounds} random rounds ({n_seekers}v{n_hiders}):")
print(f"  seeker win rate     {seeker_wins / n_rounds:.0%}")
print(f"  avg round length    {total_steps / n_rounds:.0f} steps")
print(f"  hiders spotted      {seen_steps / total_steps:.0%} of steps")
print(f"  speed               {total_steps / elapsed:,.0f} steps per second")
