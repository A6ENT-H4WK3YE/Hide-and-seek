# Quick checks that the game rules work. Run:  python test_env.py
# Uses a tiny hand-made map so every answer is easy to work out by hand.

import os
import tempfile
import numpy as np
import config as cfg
from env import HideSeekEnv, compute_visibility

TEST_MAP = """\
###########
#h..#....s#
#.........#
###########
"""
# row 1: a wall at column 4 sits between the left and right halves
# row 2: completely open


def make_env():
    path = os.path.join(tempfile.mkdtemp(), "test_map.txt")
    with open(path, "w") as f:
        f.write(TEST_MAP)
    env = HideSeekEnv(path)
    env.reset(seed=0, n_seekers=1, n_hiders=1)
    return env


def place(env, seeker, hider, t=None):
    env.seekers = np.array([seeker])
    env.hiders = np.array([hider])
    env.hider_alive[:] = True
    if t is not None:
        env.t = t
    env._update_vision()


def check(name, condition):
    print(("PASS  " if condition else "FAIL  ") + name)
    assert condition, name


env = make_env()

# --- line of sight ---
place(env, seeker=(1, 7), hider=(1, 2))
check("wall between them blocks sight", not env.seeker_sees[0, 0])

place(env, seeker=(2, 6), hider=(2, 2))
check("open row: they can see each other", env.seeker_sees[0, 0])
check("sight works both ways", env.hider_sees[0, 0])

place(env, seeker=(2, 9), hider=(2, 1))
check(f"too far apart (8 cells, range {cfg.VISION_RANGE}): not visible", not env.seeker_sees[0, 0])

# --- movement ---
place(env, seeker=(2, 9), hider=(1, 3), t=cfg.HIDING_PHASE)
env.step([0], [4])  # hider tries to move right into the wall at (1, 4)
check("walls block movement", tuple(env.hiders[0]) == (1, 3))

# --- hiding phase ---
env.reset(seed=0, n_seekers=1, n_hiders=1)
start = tuple(env.seekers[0])
env.step([3], [0])  # seeker tries to move left on step 0
check("seekers frozen during hiding phase", tuple(env.seekers[0]) == start)

# --- catching ---
env.reset(seed=0, n_seekers=1, n_hiders=1)
place(env, seeker=(2, 4), hider=(2, 2), t=cfg.HIDING_PHASE)
obs, rewards, done, info = env.step([3], [0])  # seeker steps left, now next to the hider
check("adjacent seeker catches the hider", not env.hider_alive[0])
check("round ends when all hiders are caught", done and info["winner"] == "seekers")
check("catch bonus goes to seekers", rewards["seekers"][0] > 0 and rewards["hiders"][0] < 0)

# --- rewards ---
env.reset(seed=0, n_seekers=1, n_hiders=1)
place(env, seeker=(1, 8), hider=(1, 1), t=cfg.HIDING_PHASE)
_, rewards, _, info = env.step([0], [0])
check("hidden hider gets +1, seeker -1", rewards["hiders"][0] == 1 and rewards["seekers"][0] == -1)

# --- observations ---
obs = env._get_obs()
g = obs["seekers"][0]["grid"]
check(f"observation shape is (5, {cfg.OBS_WINDOW}, {cfg.OBS_WINDOW})", g.shape == (5, cfg.OBS_WINDOW, cfg.OBS_WINDOW))
check("agent's own cell is never a wall", g[0, cfg.OBS_WINDOW // 2, cfg.OBS_WINDOW // 2] == 0)

# --- memory ---
place(env, seeker=(2, 6), hider=(2, 2), t=cfg.HIDING_PHASE)
check("memory marks a spotted hider", env.seeker_memory[0, 2, 2] == 1.0)
place(env, seeker=(1, 8), hider=(1, 1))
check("memory fades after losing sight", 0 < env.seeker_memory[0, 2, 2] < 1.0)

# --- time limit ---
env.reset(seed=0, n_seekers=1, n_hiders=1)
place(env, seeker=(1, 8), hider=(1, 1), t=cfg.MAX_STEPS - 1)
_, _, done, info = env.step([0], [0])
check("hiders win if they survive the time limit", done and info["winner"] == "hiders")

print("\nall checks passed")
