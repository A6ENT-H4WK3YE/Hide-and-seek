# Hand-written players for the v4 attack / defend game. They are fixed benchmarks (unlike self-play opponents they
# never change), and in BOT_TEAM_PROB of training games one team is played by them.
# They only use what a real team could know: what anyone on their team sees or hears (team callouts), plus a plan.
#
#   attacker bots: each round the team picks one site at random and both go there together (like a CS "execute").
#                  A bot that sees an enemy stops and keeps its aim on it (that is how players are taken out); if the
#                  team only knows where an enemy is, the bot walks towards it quietly. On a site with no enemy known
#                  they stand still and plant; once the spike is down they guard it (stay near it, looking around).
#   defender bots: the team picks one site at random and both hold it: walk to a spot on it, stand and look around.
#                  They fight like the attackers. Once the spike is down they retake together: go to it and defuse,
#                  stopping only to hold their aim on an attacker they can see.
# Routes are planned on the grid of map cells underneath, then followed by turning and walking. They run while
# travelling far and walk (silently) once close to where they are going.
#
# Usage:
#   bots = Bots(env)
#   bots.reset()                 after every env.reset()
#   bots.actions(team)           one (move, turn, switch) triple per player of that team

import numpy as np
import config as cfg
from env import MOVES, TURNS, NO_TURN, OTHER, cell_of
from geometry import line_clear, wrap_angle

SCAN_TURN = cfg.TURN_ANGLES.index(30)    # a bot holding a spot turns this way each step to look around
STEPS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
QUIET_DISTANCE = 6.0                     # walk (silent) once within this distance of where it is going


class Bots:

    def __init__(self, env, seed=0):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.dist = env.path_dist          # walking distance between every pair of floor cells
        self.site_cells = {s: [tuple(c) for c in np.argwhere(env.grid == s)] for s in "ABC" if (env.grid == s).any()}

    def reset(self):
        env = self.env
        site = self.rng.choice(sorted(self.site_cells))          # the attackers' site for this round
        self.goal = {
            "attackers": [self._pick(site) for _ in range(env.n["attackers"])],
        }
        site = self.rng.choice(sorted(self.site_cells))          # the defenders hold one site together
        self.goal["defenders"] = [self._pick(site) for _ in range(env.n["defenders"])]

    def _pick(self, site):
        y, x = self.site_cells[site][self.rng.integers(len(self.site_cells[site]))]
        return np.array([x + 0.5, y + 0.5])

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

    # ---------- one step for a whole team ----------
    def actions(self, team):
        env = self.env
        enemy = OTHER[team]
        # enemies the team knows about right now (seen or heard by anyone on the team)
        known = [j for j in range(env.n[enemy]) if env.alive[enemy][j]
                 and (env.sees[team][:, j].any() or env.hears[team][:, j].any())]
        out = []
        for i in range(env.n[team]):
            pos, heading, running = env.pos[team][i], env.heading[team][i], env.running[team][i]
            if not env.alive[team][i]:
                out.append([0, NO_TURN, 0])
                continue
            seen = [j for j in known if env.sees[team][i, j]]
            # after the plant, defenders head for the spike unless they can see an attacker right now
            retake = team == "defenders" and env.spike is not None and not seen
            if seen:
                # an enemy in view: stand still and keep the aim on it until it is out
                j = min(seen, key=lambda j: env.dist[team][i, j])
                target = env.pos[enemy][j]
                turn = self._turn_towards(heading, np.arctan2(target[1] - pos[1], target[0] - pos[0]))
                out.append([0, turn, self._switch(running, False)])
                continue
            if known and not retake:
                # the team knows where an enemy is: walk towards it quietly, looking where it is going
                j = min(known, key=lambda j: env.dist[team][i, j])
                want = self._route(pos, env.pos[enemy][j])
                turn = self._turn_towards(heading, want)
                facing = abs(wrap_angle(want - (heading + TURNS[turn]))) < np.radians(50)
                out.append([1 if facing else 0, turn, self._switch(running, False)])
                continue
            goal = self.goal[team][i] if env.spike is None else env.spike
            far = np.hypot(*(goal - pos))
            if team == "attackers" and env.spike is None and env.on_site(pos):
                out.append([0, SCAN_TURN, self._switch(running, False)])     # plant: stand still, keep looking around
                continue
            hold = 2.0 if (team == "attackers" and env.spike is not None) else 0.5
            if far < min(hold, 0.9 * cfg.DEFUSE_RANGE if team == "defenders" and env.spike is not None else hold):
                out.append([0, SCAN_TURN, self._switch(running, False)])     # hold / guard / defuse: stand, look around
                continue
            want = self._route(pos, goal)
            turn = self._turn_towards(heading, want)
            facing = abs(wrap_angle(want - (heading + TURNS[turn]))) < np.radians(50)
            out.append([1 if facing else 0, turn, self._switch(running, far > QUIET_DISTANCE)])
        return np.array(out)
