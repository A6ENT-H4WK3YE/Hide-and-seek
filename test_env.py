# Quick checks that the v4 game rules work. Run:  python test_env.py
# Uses small hand-made maps so every answer is easy to work out by hand.
# Positions are (x, y) with y pointing down; headings in degrees (0 = east, 90 = south, 180 = west).

import os
import time
import tempfile
import numpy as np
import config as cfg
from env import (GameEnv, TEAMS, NO_TURN, HIT_WALL, HIT_TEAMMATE, HIT_ENEMY, N_MINIMAP_CHANNELS, RAY_FEATURES,
                 ENEMY_FEATURES, MATE_FEATURES, VEC_DIM, MM_ME, MM_SITES, MM_ENEMIES, MM_SEARCHED)
from geometry import overlaps_wall, line_clear

TEST_MAP = """\
#############
#hh.......ss#
#...........#
#...AAA.....#
#...AAA.....#
#.....#.....#
#############
"""
# rows 1 to 4 are open; the site is the 2 x 3 block of A cells (x from 4 to 7, y from 3 to 5);
# row 5 has a wall at column 6 (x from 6 to 7, y from 5 to 6)

# two walls touching only at a corner, at the point (2, 2): (1, 1) and (2, 2) are floor on either side
CORNER_MAP = """\
#####
#s#h#
##..#
#...#
#####
"""

# actions are (move, turn, switch): switch 1 flips the walk / run mode, 0 keeps it
KEEP, SWITCH = 0, 1
STAY = [0, NO_TURN, KEEP]
FORWARD = [1, NO_TURN, KEEP]
TURN_90 = cfg.TURN_ANGLES.index(90)
GO = cfg.SETUP_STEPS          # any step from here on is after the setup phase


def make_env(text, sizes=None):
    path = os.path.join(tempfile.mkdtemp(), "test_map.txt")
    with open(path, "w") as f:
        f.write(text)
    env = GameEnv(path)
    env.reset(seed=0, sizes=sizes or {"attackers": 2, "defenders": 2})
    return env


def place(env, att, dfn, t=GO, running=True):
    # att / dfn: lists of (x, y, heading in degrees); everyone in run mode unless running=False; fresh round state
    for team, players in [("attackers", att), ("defenders", dfn)]:
        env.n[team] = len(players)
        env.pos[team] = np.array([[x, y] for x, y, _ in players], dtype=float)
        env.heading[team] = np.radians([h for _, _, h in players])
        env.alive[team] = np.ones(len(players), dtype=bool)
        env.running[team] = np.full(len(players), running)
        env.noisy[team] = np.zeros(len(players), dtype=bool)
        env.plant[team] = np.zeros(len(players), dtype=int)
    for team in TEAMS:
        other = "defenders" if team == "attackers" else "attackers"
        env.watch[team] = np.zeros((env.n[team], env.n[other]), dtype=int)
    env.offset = {"attackers": 0, "defenders": env.n["attackers"]}
    env.defuse = np.zeros(env.n["defenders"], dtype=int)
    env.spike, env.spike_timer = None, 0
    env.t, env.done, env.winner = t, False, None
    env._update_sight()
    env._update_hearing()
    env.potential = env._potential()


def act(env, att=None, dfn=None):
    # one step; anyone not given an action stays put. Returns rewards, info and the shaping part of the reward
    acts = {"attackers": att or [STAY] * env.n["attackers"], "defenders": dfn or [STAY] * env.n["defenders"]}
    old = env.potential
    obs, rewards, done, info = env.step(acts)
    shaping = cfg.GAMMA * env.potential - old
    return obs, rewards, done, info, shaping


def check(name, condition):
    print(("PASS  " if condition else "FAIL  ") + name)
    assert condition, name


def close(a, b, tol=1e-5):
    return abs(a - b) < tol


env = make_env(TEST_MAP)

# ======================================================================
# movement
# ======================================================================
place(env, att=[(11.5, 1.5, 0), (11.5, 2.5, 0)], dfn=[(4.5, 5.5, 0), (1.5, 1.5, 180)])
for _ in range(3):
    act(env, dfn=[FORWARD, STAY])     # defender runs east into the wall at x = 6
check(f"walking into a wall stops exactly one radius short (x = {env.pos['defenders'][0, 0]:.4f})",
      close(env.pos["defenders"][0, 0], 6 - cfg.AGENT_RADIUS) and close(env.pos["defenders"][0, 1], 5.5))

_, r, _, info, _ = act(env, dfn=[FORWARD, STAY])
check(f"walking into a wall costs that player {cfg.WALL_COST}, and only that player",
      env.bumped["defenders"][0] and close(r["defenders"][0] - r["defenders"][1], -cfg.WALL_COST)
      and info["bumped"]["defenders"] == 1)

place(env, att=[(11.5, 1.5, 0), (11.5, 2.5, 0)], dfn=[(8.5, 1.5, -45), (1.5, 1.5, 180)])
act(env, dfn=[FORWARD, STAY])         # diagonally up-right into the top wall: should slide along it
check(f"moving at an angle into a wall slides along it (x {env.pos['defenders'][0, 0]:.3f}, y {env.pos['defenders'][0, 1]:.3f})",
      close(env.pos["defenders"][0, 1], 1 + cfg.AGENT_RADIUS) and close(env.pos["defenders"][0, 0], 8.5 + np.cos(np.radians(45))))

place(env, att=[(11.5, 1.5, 0), (11.5, 2.5, 0)], dfn=[(2.5, 2.5, 30), (1.5, 1.5, 180)])
for _ in range(4):
    act(env, dfn=[[0, TURN_90, KEEP], STAY])
check("four 90 degree turns face the same way again", close(np.degrees(env.heading["defenders"][0]), 30))

env.reset(seed=0)
check("every round starts in walk mode", not any(env.running[t].any() for t in TEAMS))
a_start, a_head, d_start = env.pos["attackers"].copy(), env.heading["attackers"].copy(), env.pos["defenders"].copy()
act(env, att=[[1, TURN_90, SWITCH]] * 2, dfn=[FORWARD] * 2)
check("attackers frozen during the setup phase (no moving, turning or switching)",
      np.allclose(env.pos["attackers"], a_start) and np.allclose(env.heading["attackers"], a_head)
      and not env.running["attackers"].any())
check("defenders can move during the setup phase", not np.allclose(env.pos["defenders"], d_start))

place(env, att=[(11.5, 1.5, 0), (11.5, 2.5, 0)], dfn=[(2.5, 2.5, 0), (1.5, 1.5, 180)], running=False)
obs, _, _, _, _ = act(env, dfn=[FORWARD, STAY])
check(f"walking moves at {cfg.WALK_SPEED_FACTOR} x running speed, silently",
      close(env.pos["defenders"][0, 0], 2.5 + cfg.WALK_SPEED_FACTOR) and not env.noisy["defenders"][0])
act(env, dfn=[[1, NO_TURN, SWITCH], STAY])
check("switching turns on run mode (full speed, noisy)", env.running["defenders"][0] and env.noisy["defenders"][0]
      and close(env.pos["defenders"][0, 0], 4.0))

big = GameEnv("maps/three_sites.txt")
rng = np.random.default_rng(1)
bad, steps = 0, 0
for k in range(4):
    big.reset(seed=k)
    done = False
    while not done:
        acts = {t: np.stack([rng.integers(0, cfg.N_MOVES, 2), rng.integers(0, cfg.N_TURNS, 2), rng.integers(0, 2, 2)], axis=1)
                for t in TEAMS}
        _, _, done, _ = big.step(acts)
        steps += 1
        for t in TEAMS:
            for p in big.pos[t]:
                bad += overlaps_wall(big.walls, p[0], p[1], cfg.AGENT_RADIUS)
check(f"random play on three_sites never overlaps a wall ({steps} steps)", bad == 0)

# ======================================================================
# vision
# ======================================================================
mid = cfg.N_RAYS // 2
place(env, att=[(2.5, 2.5, 0), (11.5, 1.5, 0)], dfn=[(1.5, 1.5, 180), (11.5, 4.5, 0)])
check(f"straight-ahead ray stops at the east wall (distance {env.ray_dist[0, mid]:.4f}, expected 9.5)",
      close(env.ray_dist[0, mid], 9.5) and env.ray_hit[0, mid] == HIT_WALL)

place(env, att=[(9.5, 2.5, 180), (11.5, 1.5, 0)], dfn=[(3.5, 2.5, 180), (1.5, 4.5, 0)])
check("sight is one-way: the attacker sees the defender's back, the defender cannot see the attacker",
      env.sees["attackers"][0, 0] and not env.sees["defenders"][0, 0])

for angle, seen in [(40, True), (50, False)]:
    hx, hy = 2.5 + 1.5 * np.cos(np.radians(angle)), 2.5 + 1.5 * np.sin(np.radians(angle))
    place(env, att=[(2.5, 2.5, 0), (11.5, 1.5, 0)], dfn=[(hx, hy, 0), (1.5, 4.5, 0)])
    check(f"enemy {angle} degrees off to the side is {'inside' if seen else 'outside'} the 90 degree cone",
          bool(env.sees["attackers"][0, 0]) == seen)

place(env, att=[(2.5, 2.5, 0), (5.5, 2.5, 0)], dfn=[(1.5, 4.5, 0), (11.5, 4.5, 0)])
check("a ray that hits a teammate says teammate, at the edge of its body",
      env.ray_hit[0, mid] == HIT_TEAMMATE and close(env.ray_dist[0, mid], 3 - cfg.AGENT_RADIUS))
place(env, att=[(2.5, 2.5, 0), (11.5, 1.5, 0)], dfn=[(5.5, 2.5, 0), (1.5, 4.5, 0)])
check("a ray that hits an enemy says enemy", env.ray_hit[0, mid] == HIT_ENEMY)

corner = make_env(CORNER_MAP, sizes={"attackers": 1, "defenders": 1})
place(corner, att=[(1.5, 1.5, 45)], dfn=[(2.5, 2.5, 225)])
check(f"a ray exactly through a corner where two walls touch stops there (distance {corner.ray_dist[0, mid]:.4f})",
      close(corner.ray_dist[0, mid], np.sqrt(0.5)))
check("no seeing through the corner gap between two touching walls",
      not corner.sees["attackers"][0, 0] and not corner.sees["defenders"][0, 0])

big.reset(seed=0)
t0 = time.time()
for _ in range(300):
    big._update_sight()
ms = (time.time() - t0) / 300 * 1000
check(f"vision speed: {ms:.2f} ms per step for 4 players on three_sites (under 3 ms)", ms < 3.0)

# ======================================================================
# taking players out: keep them in your sights for WATCH_STEPS steps in a row
# ======================================================================
place(env, att=[(2.5, 2.5, 0), (11.5, 1.5, 90)], dfn=[(8.5, 2.5, 0), (1.5, 4.5, 180)])
for _ in range(cfg.WATCH_STEPS - 1):
    _, r, done, info, shaping = act(env)
check(f"in sight and straight ahead for {cfg.WATCH_STEPS - 1} steps: not out yet (count {env.watch['attackers'][0, 0]})",
      env.alive["defenders"][0] and env.watch["attackers"][0, 0] == cfg.WATCH_STEPS - 1)
_, r, done, info, shaping = act(env)
check(f"after {cfg.WATCH_STEPS} steps in a row: out", not env.alive["defenders"][0] and info["outs"]["attackers"] == 1)
check(f"a takedown pays the attackers +{cfg.REWARD_OUT} (zero-sum)",
      close(r["attackers"][0] - shaping, cfg.REWARD_OUT) and close(r["defenders"][0], -r["attackers"][0]))
check("a player who is out is no longer seen", not env.sees["attackers"][:, 0].any())

place(env, att=[(2.5, 2.5, 0), (11.5, 1.5, 90)], dfn=[(8.5, 2.5, 0), (1.5, 4.5, 180)])
for _ in range(cfg.WATCH_STEPS - 2):
    act(env)
act(env, att=[[0, cfg.TURN_ANGLES.index(90), KEEP], STAY])     # look away for one step
check("looking away for one step restarts the count", env.watch["attackers"][0, 0] == 0)
act(env, att=[[0, cfg.TURN_ANGLES.index(-90), KEEP], STAY])    # look back: counting from 1 again
for _ in range(cfg.WATCH_STEPS - 2):
    act(env)
check("so the defender is still in", env.alive["defenders"][0])

place(env, att=[(2.5, 2.5, -40), (11.5, 1.5, 90)], dfn=[(8.5, 2.5, 0), (1.5, 4.5, 180)])
for _ in range(cfg.WATCH_STEPS):
    act(env)
check(f"40 degrees off where the attacker looks (needs {cfg.WATCH_ANGLE}): in view but never taken out",
      env.sees["attackers"][0, 0] and env.alive["defenders"][0] and env.watch["attackers"][0, 0] == 0)

place(env, att=[(5.2, 5.5, 0), (11.5, 1.5, 90)], dfn=[(7.8, 5.5, 180), (1.5, 1.5, 180)])
for _ in range(cfg.WATCH_STEPS):
    act(env)
check("a wall in between: neither counts, neither goes out", env.watch["attackers"][0, 0] == 0
      and env.watch["defenders"][0, 0] == 0 and env.alive["defenders"][0] and env.alive["attackers"][0])

place(env, att=[(2.5, 2.5, 0), (11.5, 1.5, 90)], dfn=[(8.5, 2.5, 0), (1.5, 4.5, 180)])
for _ in range(3):
    act(env)
env.heading["defenders"][0] = np.pi            # the defender turns round 3 steps late
env._update_sight()
for _ in range(cfg.WATCH_STEPS - 3):
    act(env)
check("watching each other: whoever started first finishes first (the attacker started 3 steps earlier)",
      not env.alive["defenders"][0] and env.alive["attackers"][0])

losers = {"attackers": 0, "defenders": 0}
for k in range(40):
    place(env, att=[(2.5, 2.5, 0), (11.5, 1.5, 90)], dfn=[(8.5, 2.5, 180), (1.5, 4.5, 180)])
    env.rng = np.random.default_rng(k)
    for _ in range(cfg.WATCH_STEPS):
        act(env)
    alive = (env.alive["attackers"][0], env.alive["defenders"][0])
    assert alive in [(True, False), (False, True)], alive
    losers["defenders" if alive[0] else "attackers"] += 1
check(f"watching each other from the same moment: exactly one goes out, a coin flip (40 tries: {losers})",
      10 <= losers["attackers"] <= 30)

place(env, att=[(2.5, 2.5, 0), (11.5, 1.5, 90)], dfn=[(8.5, 2.5, 0), (1.5, 4.5, 180)], t=0)
for _ in range(cfg.SETUP_STEPS - 1):
    act(env)
check("nobody counts or goes out during the setup phase", env.watch["attackers"].max() == 0 and env.alive["defenders"][0])

# ======================================================================
# planting, spike, defusing and round end
# ======================================================================
place(env, att=[(4.5, 3.5, 0), (11.5, 1.5, 0)], dfn=[(1.5, 1.5, 180), (1.5, 2.5, 180)])
for k in range(cfg.PLANT_STEPS - 1):
    act(env)
check(f"standing still on a site plants ({env.plant['attackers'][0]}/{cfg.PLANT_STEPS}), and makes noise",
      env.plant["attackers"][0] == cfg.PLANT_STEPS - 1 and env.noisy["attackers"][0])
_, r, done, info, shaping = act(env)
check(f"after {cfg.PLANT_STEPS} steps the spike is down where the attacker stands; the round goes on",
      env.spike is not None and np.allclose(env.spike, [4.5, 3.5]) and not done and info["planted_now"])
check(f"the plant pays +{cfg.REWARD_PLANT}", close(r["attackers"][0] - shaping, cfg.REWARD_PLANT))
for _ in range(cfg.SPIKE_STEPS - 1):
    _, r, done, info, _ = act(env)
still_on = not done
_, r, done, info, _ = act(env)
check(f"the spike explodes exactly {cfg.SPIKE_STEPS} steps after the plant: attackers win",
      still_on and done and info["winner"] == "attackers")

place(env, att=[(4.5, 3.5, 0), (11.5, 1.5, 0)], dfn=[(1.5, 1.5, 180), (1.5, 2.5, 180)])
act(env)
act(env, att=[FORWARD, STAY])
check("moving resets the plant", env.plant["attackers"][0] == 0)
place(env, att=[(2.5, 2.5, 0), (11.5, 1.5, 0)], dfn=[(1.5, 1.5, 180), (1.5, 4.5, 180)])
act(env)
check("standing still off a site does not plant", env.plant["attackers"][0] == 0)

place(env, att=[(4.5, 3.5, 0), (11.5, 1.5, 0)], dfn=[(1.5, 1.5, 180), (1.5, 2.5, 180)])
for _ in range(cfg.PLANT_STEPS):
    act(env)
env.pos["defenders"][0] = [5.0, 3.5]          # a defender walks up to the spike
env.alive["attackers"][:] = False             # and every attacker is out (after the plant)
env._update_sight()
_, r, done, info, _ = act(env)
check("after the plant, every attacker out does not end the round (the spike must be defused)", not done)
for _ in range(cfg.DEFUSE_STEPS - 1):
    _, r, done, info, shaping = act(env)
check(f"a defender standing still next to the spike for {cfg.DEFUSE_STEPS} steps defuses it: defenders win",
      done and info["winner"] == "defenders" and info["defused"])
check(f"winning pays the defenders +{cfg.REWARD_WIN}", close(r["defenders"][0] + shaping, cfg.REWARD_WIN))

place(env, att=[(11.5, 1.5, 0), (11.5, 2.5, 0)], dfn=[(1.5, 1.5, 180), (1.5, 2.5, 180)], t=cfg.MAX_STEPS - 1)
_, r, done, info, _ = act(env)
check("time runs out before a plant: defenders win", done and info["winner"] == "defenders")
place(env, att=[(2.5, 2.5, 0), (11.5, 1.5, 90)], dfn=[(8.5, 2.5, 0), (1.5, 4.5, 180)])
env.alive["defenders"][1] = False
env._update_sight()
for _ in range(cfg.WATCH_STEPS):
    _, r, done, info, _ = act(env)
check("every defender out: attackers win", done and info["winner"] == "attackers")

place(env, att=[(11.5, 1.5, 0), (11.5, 2.5, 0)], dfn=[(1.5, 1.5, 180), (1.5, 2.5, 180)])
old = env.potential
act(env, att=[[1, cfg.TURN_ANGLES.index(90), KEEP], STAY])
act(env, att=[[1, NO_TURN, KEEP], STAY])
check(f"site shaping: walking towards a site raises the attackers' potential ({old:.3f} -> {env.potential:.3f})",
      env.potential > old)
_, r, _, info, _ = act(env, att=[[0, cfg.TURN_ANGLES.index(90), KEEP], STAY])
check(f"zero-sum when nobody walks into a wall, shaping included: attackers {r['attackers'][0]:+.4f}, "
      f"defenders {r['defenders'][0]:+.4f}", not any(info["bumped"].values()) and close(r["attackers"][0], -r["defenders"][0]))

# ======================================================================
# observations and team information
# ======================================================================
place(env, att=[(2.5, 2.5, 0), (2.5, 1.5, 180)], dfn=[(8.5, 2.5, 0), (11.5, 4.5, 0)])
env._update_memory()                          # what a step does after sight and hearing
obs = env._get_obs()
o = obs["attackers"][0]
check(f"shapes: rays {o['rays'].shape}, enemies {o['enemies'].shape}, mates {o['mates'].shape}, vec {o['vec'].shape}, "
      f"minimap {o['minimap'].shape}",
      o["rays"].shape == (cfg.N_RAYS, RAY_FEATURES) and o["enemies"].shape == (cfg.MAX_TEAM_SIZE, ENEMY_FEATURES)
      and o["mates"].shape == (cfg.MAX_TEAM_SIZE - 1, MATE_FEATURES) and o["vec"].shape == (VEC_DIM,)
      and o["minimap"].shape == (N_MINIMAP_CHANNELS, cfg.MINIMAP_SIZE, cfg.MINIMAP_SIZE))
check("every ray hit exactly one thing (wall, teammate or enemy)", np.allclose(o["rays"][:, 1:].sum(axis=1), 1))
check(f"attacker 0 sees the defender straight ahead, 6 away: {np.round(o['enemies'][0, :5], 3)}",
      np.allclose(o["enemies"][0, :5], [1, 1, 0, 6 / cfg.RAY_NORM, 1], atol=1e-5))
o1 = obs["attackers"][1]
check("team spotting: attacker 1 (looking the other way) still has that defender, flagged 'a teammate sees it'",
      o1["enemies"][0, 0] == 1 and o1["enemies"][0, 4] == 0 and o1["enemies"][0, 5] == 1)
check(f"teammate slot: attacker 0 is 1 away from attacker 1: {np.round(o1['mates'][0, :4], 3)}",
      o1["mates"][0, 0] == 1 and close(o1["mates"][0, 3], 1 / cfg.RAY_NORM))
check("the team's enemy memory is shared: the defender is marked on both attackers' minimaps",
      o["minimap"][MM_ENEMIES].max() == 1 and np.array_equal(o["minimap"][MM_ENEMIES], o1["minimap"][MM_ENEMIES]))
check("the minimap shows the sites and me", o["minimap"][MM_SITES].sum() > 0 and o["minimap"][MM_ME].sum() > 0)

place(env, att=[(4.5, 3.5, 0), (11.5, 1.5, 0)], dfn=[(1.5, 1.5, 180), (1.5, 2.5, 180)])
for _ in range(cfg.PLANT_STEPS):
    obs, _, _, _, _ = act(env)
v = obs["defenders"][0]["vec"]
check(f"everyone knows the spike: down, time left, direction and distance ({np.round(v[14:20], 3)})",
      v[14] == 1 and close(v[15], 1.0) and close(v[18], np.hypot(3.0, 2.0) / cfg.RAY_NORM))

# ======================================================================
# sound
# ======================================================================
check(f"hearing range scales with the map: {big.hearing_range:.2f} on three_sites (45 x 31)",
      close(big.hearing_range, cfg.HEARING_FRACTION * 45))
place(env, att=[(8.0, 5.5, 0), (11.5, 1.5, 0)], dfn=[(5.0, 5.5, 180), (1.5, 1.5, 180)])
env.hearing_range = 6.0
act(env, att=[FORWARD, STAY])                 # attacker runs east on the far side of the wall at x = 6 to 7
wall_between = not line_clear(env.walls, env.pos["defenders"][0, 0], env.pos["defenders"][0, 1],
                              env.pos["attackers"][0, 0], env.pos["attackers"][0, 1])
check("running is heard through a wall by an enemy who cannot see it",
      env.hears["defenders"][0, 0] and not env.sees["defenders"][0, 0] and wall_between)
obs = env._get_obs()
check("a heard-only enemy shows in my enemies, flagged 'only heard'",
      obs["defenders"][0]["enemies"][0, 0] == 1 and obs["defenders"][0]["enemies"][0, 6] == 1)
place(env, att=[(8.0, 5.5, 0), (11.5, 1.5, 0)], dfn=[(5.0, 5.5, 180), (1.5, 1.5, 180)], running=False)
env.hearing_range = 6.0
act(env, att=[FORWARD, STAY])
check("walking is silent", not env.hears["defenders"].any())

# ======================================================================
# team search memory
# ======================================================================
place(env, att=[(11.5, 1.5, 180), (11.5, 4.5, 0)], dfn=[(1.5, 1.5, 180), (1.5, 4.5, 180)])
env.searched = {t: np.zeros((env.H, env.W), np.float32) for t in TEAMS}
env._update_memory()
check("cells inside an attacker's cone are marked searched", env.searched["attackers"][1, 5] == 1.0)
env.heading["attackers"][0] = 0.0
values = []
for k in range(cfg.SEARCH_MEMORY_STEPS + 2):
    env._update_sight()
    env._update_memory()
    values.append(float(env.searched["attackers"][1, 5]))
steps_to_zero = values.index(0.0) + 1
check(f"searched mark is gone after exactly {cfg.SEARCH_MEMORY_STEPS} steps unseen (took {steps_to_zero})",
      steps_to_zero == cfg.SEARCH_MEMORY_STEPS)

print("\nall checks passed")
