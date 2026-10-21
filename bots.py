# Hand-written players used as fixed benchmarks. Unlike self-play opponents they never change,
# so "how do my trained agents do against the bots" is a fair way to measure progress over time.
#
#   chaser seeker:  if it sees or hears a hider, heads for it, looking at it, and catches it; if it loses
#                   track, goes to where it last saw or heard the hider; otherwise goes to random spots on
#                   the map, looking where it is going. Always runs.
#   fleeing hider:  if it sees or hears a seeker (or did in the last few steps), runs backwards to wherever
#                   is furthest along walkable paths, while still facing the seeker; otherwise stays put
#                   (silent) and slowly looks around.
#   escape hider:   a stress test for seekers that CHEATS: it always knows where every seeker is. Every step it
#                   heads for the spot furthest from both seekers among those it can reach before any of them,
#                   keeps facing the nearest seeker, walks (silent) while the seekers are out of earshot and runs
#                   once one is within hearing range or can see it. Catching it means the seekers really hunt well.
#
# The chaser and the fleeing hider only use what a real agent could know: what is inside their own cone, what they
# can hear, plus their own memory. The escape hider does not (it is an upper bound for hiders, not a fair player).
# Routes are planned on the grid of map cells underneath (shortest walking distance between cells),
# then followed in continuous space by turning and walking.
#
# Usage:
#   bots = Bots(env)
#   bots.reset()                 after every env.reset()
#   bots.seeker_actions()        one (move, turn, switch) triple per seeker
#   bots.hider_actions()         one (move, turn, switch) triple per hider
#   bots.escape_actions()        the same, for the cheating escape hider
# (switch = 1 flips the walk / run mode; the bots switch whenever their mode is not the one they want)

import numpy as np
import config as cfg
from env import MOVES, TURNS, NO_TURN, cell_of
from geometry import line_clear, wrap_angle

FLEE_MEMORY = 10      # a fleeing hider keeps running from a seeker it saw up to this many steps ago
WANDER_STEPS = 30     # a wandering seeker picks a new random spot after this many steps, even if not there yet
SCAN_TURN = cfg.TURN_ANGLES.index(30)    # a waiting hider turns this way each step to look around
STEPS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WALK, RUN = False, True


class Bots:

    def __init__(self, env, seed=0):
        self.env = env
        self.rng = np.random.default_rng(seed)
        # shortest walking distance between every pair of floor cells (the game works it out once per map)
        self.floor = list(env.path_dist.keys())
        self.dist = env.path_dist

    def reset(self):
        self.seeker_target = [None] * self.env.n_seekers     # where each seeker last saw or heard a hider
        self.seeker_wander = [None] * self.env.n_seekers     # (random spot, steps spent going there)
        self.hider_threat = [None] * self.env.n_hiders       # (position, steps ago) of the last seeker seen or heard

    # ---------- steering ----------
    def _switch(self, running, want):
        # 1 if the walk / run mode has to flip to get the mode we want, otherwise 0 (keep)
        return int(bool(running) != want)

    def _route(self, pos, target):
        # direction to walk from pos towards target: straight there if the line is clear,
        # otherwise towards the centre of the next cell on the shortest grid path
        if line_clear(self.env.walls, pos[0], pos[1], target[0], target[1]):
            goal = target
        else:
            here = cell_of(pos)
            d = self.dist[cell_of(target)]
            best = min(self._neighbours(here), key=lambda c: d[c])
            goal = (best[1] + 0.5, best[0] + 0.5)
        return np.arctan2(goal[1] - pos[1], goal[0] - pos[0])

    def _neighbours(self, cell):
        y, x = cell
        cells = [cell]
        for dy, dx in STEPS4:
            if not self.env.walls[y + dy, x + dx]:
                cells.append((y + dy, x + dx))
        return cells

    def _turn_towards(self, heading, angle):
        # the turn action that leaves the agent looking closest to angle
        return int(np.argmin(np.abs(wrap_angle(angle - (heading + TURNS)))))

    def _move_towards(self, heading, angle):
        # the move (forward, back, strafe left / right) closest to the wanted direction
        best, best_dot = 0, -2.0
        for m, (fwd, right) in MOVES.items():
            if m == 0:
                continue
            move_angle = heading + np.arctan2(right, fwd)
            dot = np.cos(wrap_angle(angle - move_angle))
            if dot > best_dot:
                best, best_dot = m, dot
        return best

    # ---------- seekers ----------
    def seeker_actions(self):
        env = self.env
        actions = []
        for i in range(env.n_seekers):
            pos, heading = env.seekers[i], env.seeker_heading[i]
            visible = [env.hiders[j] for j in np.where(env.seeker_sees[i])[0]]
            heard = [env.hiders[j] for j in np.where(env.seeker_hears[i] & env.hider_alive)[0]]
            if visible or heard:
                self.seeker_target[i] = min(visible or heard, key=lambda h: np.hypot(*(h - pos))).copy()
                self.seeker_wander[i] = None
            elif self.seeker_target[i] is not None and np.hypot(*(self.seeker_target[i] - pos)) < 0.5:
                self.seeker_target[i] = None      # reached the last known spot, nobody there

            target = self.seeker_target[i]
            if target is None:
                # wander: walk to a random spot, pick a new one when there (or after a while)
                spot = self.seeker_wander[i]
                if spot is None or spot[1] >= WANDER_STEPS or np.hypot(*(spot[0] - pos)) < 0.5:
                    y, x = self.floor[self.rng.integers(len(self.floor))]
                    spot = (np.array([x + 0.5, y + 0.5]), 0)
                self.seeker_wander[i] = (spot[0], spot[1] + 1)
                target = spot[0]

            want = self._route(pos, target)
            turn = self._turn_towards(heading, want)
            new_heading = heading + TURNS[turn]
            close_enough = visible and np.hypot(*(target - pos)) <= 0.9 * cfg.CATCH_DISTANCE
            facing = abs(wrap_angle(want - new_heading)) < np.radians(50)
            move = 1 if (facing and not close_enough) else 0
            actions.append([move, turn, self._switch(env.seeker_running[i], RUN)])
        return np.array(actions)

    # ---------- hiders ----------
    def hider_actions(self):
        env = self.env
        actions = []
        for j in range(env.n_hiders):
            pos, heading = env.hiders[j], env.hider_heading[j]
            if not env.hider_alive[j]:
                actions.append([0, NO_TURN, 0])
                continue
            visible = [env.seekers[i] for i in np.where(env.hider_sees[j] | env.hider_hears[j])[0]]
            if visible:
                self.hider_threat[j] = (min(visible, key=lambda s: np.hypot(*(s - pos))).copy(), 0)
            elif self.hider_threat[j] is not None:
                spot, age = self.hider_threat[j]
                self.hider_threat[j] = (spot, age + 1) if age + 1 <= FLEE_MEMORY else None

            if self.hider_threat[j] is None:
                actions.append([0, SCAN_TURN, self._switch(env.hider_running[j], WALK)])   # nothing to run from: stay (silent) and look around
                continue
            threat = self.hider_threat[j][0]
            # head for the neighbouring cell that is furthest (walking) from the threat
            d = self.dist[cell_of(threat)]
            best = max(self._neighbours(cell_of(pos)), key=lambda c: d[c])
            goal = np.array([best[1] + 0.5, best[0] + 0.5])
            away = np.arctan2(goal[1] - pos[1], goal[0] - pos[0])
            # keep looking at the threat while backing away
            turn = self._turn_towards(heading, np.arctan2(threat[1] - pos[1], threat[0] - pos[0]))
            move = self._move_towards(heading + TURNS[turn], away) if np.hypot(*(goal - pos)) > 0.1 else 0
            actions.append([move, turn, self._switch(env.hider_running[j], RUN)])
        return np.array(actions)

    # ---------- escape hider (cheats: knows where every seeker is) ----------
    def escape_actions(self):
        env = self.env
        # walking distance from every cell to the nearest seeker
        danger = np.minimum.reduce([self.dist[cell_of(p)] for p in env.seekers])
        actions = []
        for j in range(env.n_hiders):
            pos, heading = env.hiders[j], env.hider_heading[j]
            if not env.hider_alive[j]:
                actions.append([0, NO_TURN, 0])
                continue
            mine = self.dist[cell_of(pos)]
            # cells I reach before any seeker; of those, the one furthest from the seekers (nearest to me on a tie)
            safe = (mine < danger) & ~env.walls
            score = np.where(safe if safe.any() else ~env.walls, danger - 0.01 * mine, -np.inf)
            y, x = np.unravel_index(np.argmax(score), score.shape)
            target = np.array([x + 0.5, y + 0.5])
            want = self._route(pos, target)
            nearest = min(env.seekers, key=lambda p: np.hypot(*(p - pos)))
            near_d = np.hypot(*(nearest - pos))
            # watch the nearest seeker while moving
            turn = self._turn_towards(heading, np.arctan2(nearest[1] - pos[1], nearest[0] - pos[0]))
            move = self._move_towards(heading + TURNS[turn], want) if np.hypot(*(target - pos)) > 0.3 else 0
            # quiet while out of earshot, sprint once a seeker could hear or see it
            run = near_d < env.hearing_range + 2 or bool(env.seeker_sees[:, j].any())
            actions.append([move, turn, self._switch(env.hider_running[j], run)])
        return np.array(actions)
