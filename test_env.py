# Quick checks that the game rules work. Run:  python test_env.py
# Uses a tiny hand-made map so every answer is easy to work out by hand.

import os
import tempfile
import numpy as np
import config as cfg
from env import HideSeekEnv, compute_visibility, N_CHANNELS, N_MINIMAP_CHANNELS, CH_SEARCHED, MM_ME, MM_TEAMMATES, MM_SEARCHED

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
expected = cfg.REWARD_SEEN - cfg.REWARD_CATCH - cfg.REWARD_WIN   # -1 - 5 - 10 = -16 for hiders
check(f"final catch: hiders get seen + catch + loss = {expected}", rewards["hiders"][0] == expected)
check("zero-sum: seekers get exactly minus the hiders", rewards["seekers"][0] == -rewards["hiders"][0])

# --- rewards ---
env.reset(seed=0, n_seekers=1, n_hiders=1)
place(env, seeker=(1, 8), hider=(1, 1), t=cfg.HIDING_PHASE)
_, rewards, _, info = env.step([0], [0])
check("hidden hider gets +1, seeker -1", rewards["hiders"][0] == 1 and rewards["seekers"][0] == -1)

# --- observations ---
obs = env._get_obs()
g = obs["seekers"][0]["grid"]
check(f"close-up shape is ({N_CHANNELS}, {cfg.OBS_WINDOW}, {cfg.OBS_WINDOW})", g.shape == (N_CHANNELS, cfg.OBS_WINDOW, cfg.OBS_WINDOW))
mm = obs["seekers"][0]["minimap"]
check(f"minimap shape is ({N_MINIMAP_CHANNELS}, {cfg.MINIMAP_SIZE}, {cfg.MINIMAP_SIZE})", mm.shape == (N_MINIMAP_CHANNELS, cfg.MINIMAP_SIZE, cfg.MINIMAP_SIZE))
check("minimap marks my own position", mm[MM_ME].sum() > 0)
check("agent's own cell is never a wall", g[0, cfg.OBS_WINDOW // 2, cfg.OBS_WINDOW // 2] == 0)

# --- memory ---
place(env, seeker=(2, 6), hider=(2, 2), t=cfg.HIDING_PHASE)
check("memory marks a spotted hider", env.seeker_memory[0, 2, 2] == 1.0)
place(env, seeker=(1, 8), hider=(1, 1))
check("memory fades after losing sight", 0 < env.seeker_memory[0, 2, 2] < 1.0)

# --- time limit ---
env.reset(seed=0, n_seekers=1, n_hiders=1)
place(env, seeker=(1, 8), hider=(1, 1), t=cfg.MAX_STEPS - 1)
_, rewards, done, info = env.step([0], [0])
check("hiders win if they survive the time limit", done and info["winner"] == "hiders")
expected = cfg.REWARD_HIDDEN + cfg.REWARD_WIN   # +1 + 10 = 11 for hiders
check(f"surviving the last step: hiders get hidden + win = {expected}", rewards["hiders"][0] == expected)

# --- shared team search memory ---
env.reset(seed=0, n_seekers=2, n_hiders=1)
env.seekers = np.array([[2, 6], [1, 8]])     # seeker 0 looks along the open row 2, seeker 1 sits in the right corner
env.hiders = np.array([[1, 1]])
env.hider_alive[:] = True
env.t = cfg.HIDING_PHASE
env._update_vision()
obs = env._get_obs()
check("a cell seeker 0 sees is marked as searched", env.searched["seekers"][2, 3] == 1.0)
check("seeker 1 can't see that cell itself", not env.view_of(env.seekers[1])[2, 3])
r = cfg.OBS_WINDOW // 2
in_window = obs["seekers"][1]["grid"][CH_SEARCHED, r + (2 - 1), r + (3 - 8)]   # cell (2, 3) seen from (1, 8)
check("but it shows as searched in seeker 1's own observation (shared by the team)", in_window == 1.0)
check("hiders have their own separate searched layer", env.searched["hiders"][2, 3] != env.searched["seekers"][2, 3] or
      env.view_of(env.hiders[0])[2, 3])
check("teammate appears on the minimap", obs["seekers"][0]["minimap"][MM_TEAMMATES].sum() > 0)

# move both seekers away so (2, 3) is out of everyone's sight, then count the steps until it fades out
env.seekers = np.array([[1, 8], [1, 9]])
env.hiders = np.array([[1, 1]])
values = []
for k in range(cfg.SEARCH_MEMORY_STEPS + 2):
    env._update_vision()
    values.append(float(env.searched["seekers"][2, 3]))
check("the cell is now out of every seeker's sight", not env.view_of(env.seekers[0])[2, 3] and not env.view_of(env.seekers[1])[2, 3])
check(f"searched mark fades steadily ({values[0]:.1f} -> {values[1]:.1f} -> {values[2]:.1f} ...)", values[0] > values[1] > values[2] > 0)
steps_to_zero = values.index(0.0) + 1
check(f"searched mark is gone after exactly {cfg.SEARCH_MEMORY_STEPS} steps unseen (took {steps_to_zero})",
      steps_to_zero == cfg.SEARCH_MEMORY_STEPS)

print("\nall checks passed")
