# Quick checks that the game rules work. Run:  python test_env.py
# Uses tiny hand-made maps so every answer is easy to work out by hand.
# Positions are (x, y) with y pointing down; headings in degrees (0 = east, 90 = south, 180 = west).

import os
import time
import tempfile
import numpy as np
import config as cfg
from env import (HideSeekEnv, NO_TURN, HIT_WALL, HIT_TEAMMATE, HIT_ENEMY, N_MINIMAP_CHANNELS,
                 RAY_FEATURES, ENEMY_FEATURES, VEC_DIM, MM_ME, MM_SEARCHED)
from geometry import overlaps_wall

TEST_MAP = """\
###########
#h..#....s#
#.........#
#.........#
###########
"""
# row 1: a wall at column 4 (x from 4 to 5, y from 1 to 2) splits the top row in two
# rows 2 and 3: completely open

# two walls touching only at a corner, at the point (2, 2): (1, 1) and (2, 2) are floor on either side
CORNER_MAP = """\
#####
#s#h#
##..#
#...#
#####
"""

# SOUND_MAP: a wall down the middle of the top part, so two cells can be close in a straight line but far to walk
SOUND_MAP = """\
###########
#h...#...s#
#....#....#
#....#....#
#....#....#
#.........#
###########
"""
# (1, 4) and (1, 6) are 2 apart in a straight line, but 10 steps apart walking round the wall via row 5

# actions are (move, turn, switch): switch 1 flips the walk / run mode, 0 keeps it.
# place() below puts agents in run mode unless told otherwise (most checks are about full-speed moves).
KEEP, SWITCH = 0, 1
STAY = [0, NO_TURN, KEEP]
FORWARD = [1, NO_TURN, KEEP]
TURN_90 = cfg.TURN_ANGLES.index(90)


def make_env(text, n_seekers=1, n_hiders=1):
    path = os.path.join(tempfile.mkdtemp(), "test_map.txt")
    with open(path, "w") as f:
        f.write(text)
    env = HideSeekEnv(path)
    env.reset(seed=0, n_seekers=n_seekers, n_hiders=n_hiders)
    return env


def place(env, seekers, hiders, t=None, running=True):
    # seekers / hiders: lists of (x, y, heading in degrees); everyone in run mode (or walk mode if running=False)
    env.seekers = np.array([[x, y] for x, y, _ in seekers], dtype=float)
    env.seeker_heading = np.radians([h for _, _, h in seekers])
    env.hiders = np.array([[x, y] for x, y, _ in hiders], dtype=float)
    env.hider_heading = np.radians([h for _, _, h in hiders])
    env.n_seekers, env.n_hiders = len(seekers), len(hiders)
    env.hider_alive = np.ones(len(hiders), dtype=bool)
    env.seeker_memory = np.zeros((len(seekers), env.H, env.W), dtype=np.float32)
    env.hider_memory = np.zeros((len(hiders), env.H, env.W), dtype=np.float32)
    env.seeker_noisy = np.zeros(len(seekers), dtype=bool)
    env.hider_noisy = np.zeros(len(hiders), dtype=bool)
    env.seeker_running = np.full(len(seekers), running)
    env.hider_running = np.full(len(hiders), running)
    env.done, env.winner = False, None
    if t is not None:
        env.t = t
    env._update_sight()
    env._update_hearing()


def random_actions(rng, n):
    return np.stack([rng.integers(0, cfg.N_MOVES, n), rng.integers(0, cfg.N_TURNS, n), rng.integers(0, cfg.N_SPEEDS, n)], axis=1)


def check(name, condition):
    print(("PASS  " if condition else "FAIL  ") + name)
    assert condition, name


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


env = make_env(TEST_MAP)
HUNT = cfg.HIDING_PHASE   # any step from here on is the hunting phase

# ======================================================================
# movement
# ======================================================================
place(env, seekers=[(8.5, 3.5, 0)], hiders=[(2.5, 1.5, 0)], t=HUNT)
for _ in range(3):
    env.step([STAY], [FORWARD])     # hider walks east into the wall at x = 4
check(f"walking into a wall stops exactly one radius short (x = {env.hiders[0, 0]:.4f}, expected {4 - cfg.AGENT_RADIUS})",
      close(env.hiders[0, 0], 4 - cfg.AGENT_RADIUS) and close(env.hiders[0, 1], 1.5))

place(env, seekers=[(8.5, 3.5, 0)], hiders=[(6.5, 1.5, -45)], t=HUNT)
env.step([STAY], [FORWARD])         # diagonally up-right into the top wall: should slide along it
expected_x = 6.5 + cfg.MOVE_SPEED * np.cos(np.radians(45))
check(f"moving at an angle into a wall slides along it (x {env.hiders[0, 0]:.3f}, y {env.hiders[0, 1]:.3f})",
      close(env.hiders[0, 1], 1 + cfg.AGENT_RADIUS) and close(env.hiders[0, 0], expected_x))

place(env, seekers=[(8.5, 3.5, 0)], hiders=[(2.5, 2.5, 30)], t=HUNT)
for _ in range(4):
    env.step([STAY], [[0, TURN_90, 0]])
check("four 90 degree turns face the same way again", close(np.degrees(env.hider_heading[0]), 30))

env.reset(seed=0, n_seekers=1, n_hiders=1)
start, start_heading = env.seekers[0].copy(), env.seeker_heading[0]
check("every round starts in walk mode", not env.seeker_running.any() and not env.hider_running.any())
env.step([[1, TURN_90, SWITCH]], [STAY])
check("seekers frozen during hiding phase (no moving, no turning, no switching to run)",
      np.allclose(env.seekers[0], start) and env.seeker_heading[0] == start_heading and not env.seeker_running[0])

tiny = HideSeekEnv("maps/tiny.txt")
rng = np.random.default_rng(1)
bad, steps = 0, 0
for k in range(20):
    tiny.reset(seed=k, n_seekers=2, n_hiders=1)
    done = False
    while not done:
        _, _, done, _ = tiny.step(random_actions(rng, 2), random_actions(rng, 1))
        steps += 1
        for p in np.concatenate([tiny.seekers, tiny.hiders]):
            bad += overlaps_wall(tiny.walls, p[0], p[1], cfg.AGENT_RADIUS)
check(f"random play on the tiny map never overlaps a wall ({steps} steps)", bad == 0)

place(env, seekers=[(8.5, 3.5, 0)], hiders=[(2.5, 2.5, 0)], t=HUNT, running=False)
obs, _, _, _ = env.step([STAY], [FORWARD])
check(f"walking moves at {cfg.WALK_SPEED_FACTOR} x running speed (x {env.hiders[0, 0]:.2f})",
      close(env.hiders[0, 0], 2.5 + cfg.WALK_SPEED_FACTOR * cfg.MOVE_SPEED))
check("the agent can see its own mode (last number in vec: 0 = walking)", obs["hiders"][0]["vec"][-1] == 0.0)
obs, _, _, _ = env.step([STAY], [[1, NO_TURN, SWITCH]])
check(f"switching turns on run mode, at full speed straight away (x {env.hiders[0, 0]:.2f})",
      env.hider_running[0] and close(env.hiders[0, 0], 3.0 + cfg.MOVE_SPEED) and obs["hiders"][0]["vec"][-1] == 1.0)
env.step([STAY], [FORWARD])
check("run mode stays on until switched again", env.hider_running[0] and close(env.hiders[0, 0], 5.0))
env.step([STAY], [[0, NO_TURN, SWITCH]])
check("switching again goes back to walking", not env.hider_running[0])

# ======================================================================
# vision
# ======================================================================
mid = cfg.N_RAYS // 2   # the ray straight ahead
place(env, seekers=[(2.5, 2.5, 0)], hiders=[(1.5, 1.5, 0)])
check(f"straight-ahead ray stops at the east wall (distance {env.ray_dist[0, mid]:.4f}, expected 7.5)",
      close(env.ray_dist[0, mid], 7.5) and env.ray_hit[0, mid] == HIT_WALL)

place(env, seekers=[(7.5, 1.5, 180)], hiders=[(2.5, 1.5, 0)])
check("wall between them blocks sight", not env.seeker_sees[0, 0])

place(env, seekers=[(7.5, 2.5, 180)], hiders=[(2.5, 2.5, 0)])
check("open row, facing each other: both see each other", env.seeker_sees[0, 0] and env.hider_sees[0, 0])

place(env, seekers=[(7.5, 2.5, 0)], hiders=[(2.5, 2.5, 0)])
check("hider behind the seeker (outside the cone): not seen", not env.seeker_sees[0, 0])

place(env, seekers=[(7.5, 2.5, 180)], hiders=[(2.5, 2.5, 180)])
check("sight is one-way: the seeker sees the hider's back, the hider cannot see the seeker",
      env.seeker_sees[0, 0] and not env.hider_sees[0, 0])

for angle, seen in [(40, True), (50, False)]:
    hx, hy = 2.5 + 1.8 * np.cos(np.radians(angle)), 2.5 + 1.8 * np.sin(np.radians(angle))
    place(env, seekers=[(2.5, 2.5, 0)], hiders=[(hx, hy, 0)])
    check(f"hider {angle} degrees off to the side is {'inside' if seen else 'outside'} the 90 degree cone",
          bool(env.seeker_sees[0, 0]) == seen)

place(env, seekers=[(2.5, 2.5, 0), (5.5, 2.5, 0)], hiders=[(1.5, 1.5, 0)])
check(f"a ray that hits a teammate says teammate, at the edge of its body (distance {env.ray_dist[0, mid]:.2f})",
      env.ray_hit[0, mid] == HIT_TEAMMATE and close(env.ray_dist[0, mid], 3 - cfg.AGENT_RADIUS))
place(env, seekers=[(2.5, 2.5, 0)], hiders=[(5.5, 2.5, 0)])
check("a ray that hits an enemy says enemy", env.ray_hit[0, mid] == HIT_ENEMY)

corner = make_env(CORNER_MAP)
place(corner, seekers=[(1.5, 1.5, 45)], hiders=[(2.5, 2.5, 225)])
check(f"a ray exactly through a corner where two walls touch stops there (distance {corner.ray_dist[0, mid]:.4f})",
      close(corner.ray_dist[0, mid], np.sqrt(0.5)))
check("no seeing through the corner gap between two touching walls",
      not corner.seeker_sees[0, 0] and not corner.hider_sees[0, 0])

tiny.reset(seed=0, n_seekers=2, n_hiders=1)
t0 = time.time()
for _ in range(1000):
    tiny._update_sight()
ms = (time.time() - t0)
check(f"vision speed: {ms:.2f} ms per step for 3 agents on the tiny map (under 2 ms)", ms < 2.0)

# ======================================================================
# catching
# ======================================================================
place(env, seekers=[(3.2, 2.5, 180)], hiders=[(2.5, 2.5, 0)], t=HUNT)
_, rewards, done, info = env.step([STAY], [STAY])
check("close, facing the hider, nothing in between: caught", not env.hider_alive[0])
check("round ends when all hiders are caught", done and info["winner"] == "seekers")
closeness = 1 - 0.7 / cfg.CLOSENESS_RANGE
time_left = (cfg.MAX_STEPS - (HUNT + 1)) / (cfg.MAX_STEPS - cfg.HIDING_PHASE)
expected = cfg.REWARD_SEEN * closeness + cfg.REWARD_CATCH + cfg.REWARD_WIN + cfg.REWARD_TIME_BONUS * time_left
check(f"catch: seekers get seen x closeness + catch + win + time bonus = {expected:.3f}",
      close(rewards["seekers"][0], expected, 1e-4))
check("zero-sum: hiders get exactly minus the seekers", rewards["hiders"][0] == -rewards["seekers"][0])

place(env, seekers=[(3.2, 2.5, 0)], hiders=[(2.5, 2.5, 0)], t=HUNT)
env.step([STAY], [STAY])
check("close but facing away: no catch", env.hider_alive[0])

place(env, seekers=[(3.2, 2.5, 140)], hiders=[(2.5, 2.5, 0)], t=HUNT)
env.step([STAY], [STAY])
check(f"close, hider 40 degrees off where the seeker looks: seen but not caught (needs {cfg.CATCH_ANGLE})",
      env.hider_alive[0] and env.seeker_sees[0, 0])

place(env, seekers=[(3.4, 2.5, 180)], hiders=[(2.5, 2.5, 0)], t=HUNT)
env.step([STAY], [STAY])
check(f"facing the hider but 0.9 away (catch distance {cfg.CATCH_DISTANCE}): no catch", env.hider_alive[0])

# ======================================================================
# rewards
# ======================================================================
env.reset(seed=0, n_seekers=1, n_hiders=1)
_, rewards, _, _ = env.step([STAY], [STAY])
check("hiding phase: no rewards", rewards["seekers"][0] == 0 and rewards["hiders"][0] == 0)

place(env, seekers=[(8.5, 1.5, 0)], hiders=[(1.5, 1.5, 0)], t=HUNT)
_, rewards, _, _ = env.step([STAY], [STAY])
check(f"no hider in sight: seekers {cfg.REWARD_UNSEEN}, hiders {-cfg.REWARD_UNSEEN}",
      close(rewards["seekers"][0], cfg.REWARD_UNSEEN) and close(rewards["hiders"][0], -cfg.REWARD_UNSEEN))

place(env, seekers=[(7.5, 2.5, 180)], hiders=[(2.5, 2.5, 0)], t=HUNT)
_, rewards, _, _ = env.step([STAY], [STAY])
expected = cfg.REWARD_SEEN * (1 - 5 / cfg.CLOSENESS_RANGE)
check(f"hider in sight 5 away: seekers get {expected:.3f} (more when closer)", close(rewards["seekers"][0], expected, 1e-5))

place(env, seekers=[(8.5, 1.5, 0)], hiders=[(1.5, 1.5, 0)], t=cfg.MAX_STEPS - 1)
_, rewards, done, info = env.step([STAY], [STAY])
expected = cfg.REWARD_UNSEEN + cfg.REWARD_LOSS
check(f"hiders survive the time limit: hiders win, seekers get {expected}",
      done and info["winner"] == "hiders" and close(rewards["seekers"][0], expected, 1e-5))

# the farming exploit from the grid version: is following a hider without catching it ever better?
# Worst case: watching from 0.9 away, just outside catch distance, for the whole round.
place(env, seekers=[(3.4, 2.5, 180)], hiders=[(2.5, 2.5, 0)], t=HUNT)
follow, done = 0.0, False
while not done:
    _, rewards, done, _ = env.step([STAY], [STAY])
    follow += rewards["seekers"][0]
place(env, seekers=[(3.2, 2.5, 180)], hiders=[(2.5, 2.5, 0)], t=HUNT)
_, rewards, _, _ = env.step([STAY], [STAY])
catch = rewards["seekers"][0]
check(f"no farming: watching a hider from 0.9 away for the whole round ({follow:+.1f}) pays less than catching it ({catch:+.1f})",
      catch > follow)

# ======================================================================
# observations
# ======================================================================
place(env, seekers=[(7.5, 2.5, 180)], hiders=[(2.5, 2.5, 180)], t=HUNT)
obs = env._get_obs()
o = obs["seekers"][0]
check(f"shapes: rays {o['rays'].shape}, enemies {o['enemies'].shape}, heard {o['heard'].shape}, vec {o['vec'].shape}, minimap {o['minimap'].shape}",
      o["rays"].shape == (cfg.N_RAYS, RAY_FEATURES) and o["enemies"].shape == (cfg.MAX_TEAM_SIZE, ENEMY_FEATURES)
      and o["heard"].shape == (cfg.MAX_TEAM_SIZE, ENEMY_FEATURES)
      and o["vec"].shape == (VEC_DIM,) and o["minimap"].shape == (N_MINIMAP_CHANNELS, cfg.MINIMAP_SIZE, cfg.MINIMAP_SIZE))
check("every ray hit exactly one thing (wall, teammate or enemy)", np.allclose(o["rays"][:, 1:].sum(axis=1), 1))
check(f"seen hider is in the first enemy slot, straight ahead, 5 away: {np.round(o['enemies'][0], 3)}",
      np.allclose(o["enemies"][0], [1, 1, 0, 5 / cfg.RAY_NORM], atol=1e-5) and not o["enemies"][1:].any())
check("the hider looking the other way has an empty enemy list", not obs["hiders"][0]["enemies"].any())
check("minimap marks my own position", o["minimap"][MM_ME].sum() > 0)

# ======================================================================
# sound
# ======================================================================
snd = make_env(SOUND_MAP)
# seeker at the bottom right looking east (away), so it can only hear, not see
place(snd, seekers=[(7.5, 5.5, 0)], hiders=[(4.5, 5.5, 180)], t=HUNT)
obs, _, _, _ = snd.step([STAY], [FORWARD])     # hider runs west, ends 4 steps' walk from the seeker
check(f"a running hider makes noise", snd.hider_noisy[0])
check(f"the seeker hears it (walking distance {snd.sound_dist[0, 0]:.0f}, range {cfg.HEARING_RANGE})",
      snd.seeker_hears[0, 0] and not snd.seeker_sees[0, 0])
check(f"heard slot: behind the seeker, walking distance 4: {np.round(obs['seekers'][0]['heard'][0], 3)}",
      np.allclose(obs["seekers"][0]["heard"][0], [1, -1, 0, 4 / cfg.RAY_NORM], atol=1e-5))
check("a heard enemy is marked in the seeker's memory", snd.seeker_memory[0, 5, 3] == 1.0)

place(snd, seekers=[(7.5, 5.5, 0)], hiders=[(4.5, 5.5, 180)], t=HUNT, running=False)
obs, _, _, _ = snd.step([STAY], [FORWARD])
check("walking is silent: not heard", not snd.hider_noisy[0] and not snd.seeker_hears[0, 0]
      and not obs["seekers"][0]["heard"].any())

place(snd, seekers=[(7.5, 5.5, 0)], hiders=[(4.5, 5.5, 180)], t=HUNT)
snd.step([STAY], [STAY])
check("running on the spot (not moving) makes no noise", not snd.hider_noisy[0])

place(snd, seekers=[(6.5, 1.5, 0)], hiders=[(4.5, 1.5, 90)], t=HUNT)
snd.step([STAY], [FORWARD])                    # 2 apart in a straight line, but the wall is between them
check(f"sound does not go through walls: 2 apart in a straight line, {snd.sound_dist[0, 0]:.0f} to walk, not heard",
      snd.hider_noisy[0] and not snd.seeker_hears[0, 0])

place(snd, seekers=[(4.5, 5.5, 180)], hiders=[(7.5, 5.5, 0)], t=HUNT)
snd.step([FORWARD], [STAY])
check("hiders hear running seekers too", snd.hider_hears[0, 0])

place(snd, seekers=[(7.5, 5.5, 0), (8.5, 5.5, 0)], hiders=[(1.5, 1.5, 0)], t=HUNT)
obs, _, _, _ = snd.step([FORWARD, STAY], [STAY])
check("teammates' footsteps are not reported as heard enemies", not obs["seekers"][1]["heard"].any())

snd.reset(seed=0, n_seekers=1, n_hiders=1)
snd.step([[1, NO_TURN, SWITCH]], [STAY])
check("seekers frozen in the hiding phase make no noise", not snd.seeker_noisy[0])

# ======================================================================
# shared team search memory
# ======================================================================
env.reset(seed=0, n_seekers=2, n_hiders=1)
place(env, seekers=[(8.5, 1.5, 180), (8.5, 3.5, 0)], hiders=[(1.5, 3.5, 270)], t=HUNT)
env.searched = {"seekers": np.zeros((env.H, env.W), np.float32), "hiders": np.zeros((env.H, env.W), np.float32)}
env._update_memory()
s = env.searched["seekers"]
check("cells inside seeker 0's cone are marked searched", s[1, 5] == 1.0 and s[2, 3] == 1.0)
check("a cell hidden behind the wall is not", s[1, 2] == 0.0)
check("a cell behind the seeker's back is not", s[1, 9] == 0.0)
obs = env._get_obs()
check("both seekers get the same searched layer on their minimap (shared by the team)",
      np.array_equal(obs["seekers"][0]["minimap"][MM_SEARCHED], obs["seekers"][1]["minimap"][MM_SEARCHED]))

# turn seeker 0 to face the east wall, so (1, 5) is out of every seeker's sight, then count the steps until it fades out
env.seeker_heading[0] = 0.0
values = []
for k in range(cfg.SEARCH_MEMORY_STEPS + 2):
    env._update_sight()
    env._update_memory()
    values.append(float(env.searched["seekers"][1, 5]))
check(f"searched mark fades steadily ({values[0]:.1f} -> {values[1]:.1f} -> {values[2]:.1f} ...)", values[0] > values[1] > values[2] > 0)
steps_to_zero = values.index(0.0) + 1
check(f"searched mark is gone after exactly {cfg.SEARCH_MEMORY_STEPS} steps unseen (took {steps_to_zero})",
      steps_to_zero == cfg.SEARCH_MEMORY_STEPS)

print("\nall checks passed")
