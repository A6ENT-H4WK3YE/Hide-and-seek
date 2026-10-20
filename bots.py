# Hand-written players used as fixed benchmarks. Unlike self-play opponents they never change,
# so "how do my trained agents do against the bots" is a fair way to measure progress over time.
#
#   chaser seeker:  if it sees a hider, walks the shortest path towards it; if it loses sight,
#                   goes to where it last saw the hider; otherwise wanders randomly.
#   fleeing hider:  if it sees a seeker (or saw one in the last few steps), moves to the cell that
#                   is furthest away along walkable paths; otherwise stays still.
#
# Both only use what a real agent could know: what is in their own line of sight, plus their own memory.
#
# Usage:
#   bots = Bots(env)
#   bots.reset()                 after every env.reset()
#   bots.seeker_actions()        one action per seeker
#   bots.hider_actions()         one action per hider

from collections import deque
import numpy as np
from env import MOVES

FLEE_MEMORY = 10    # a fleeing hider keeps running from a seeker it saw up to this many steps ago


class Bots:

    def __init__(self, env, seed=0):
        self.env = env
        self.rng = np.random.default_rng(seed)
        # shortest walking distance between every pair of floor cells, worked out once per map
        H, W = env.H, env.W
        self.dist = {}
        for y in range(H):
            for x in range(W):
                if not env.walls[y, x]:
                    self.dist[(y, x)] = self._bfs((y, x))

    def _bfs(self, src):
        env = self.env
        d = np.full((env.H, env.W), 10_000)
        d[src] = 0
        queue = deque([src])
        while queue:
            y, x = queue.popleft()
            for dy, dx in MOVES.values():
                ny, nx = y + dy, x + dx
                if 0 <= ny < env.H and 0 <= nx < env.W and not env.walls[ny, nx] and d[ny, nx] == 10_000:
                    d[ny, nx] = d[y, x] + 1
                    queue.append((ny, nx))
        return d

    def reset(self):
        self.seeker_target = [None] * self.env.n_seekers     # where each seeker last saw a hider
        self.hider_threat = [None] * self.env.n_hiders       # (position, steps ago) of the last seeker seen

    def _best_move(self, pos, score):
        # try every action, keep the one(s) with the highest score, break ties randomly
        scores = []
        for a, (dy, dx) in MOVES.items():
            y, x = pos[0] + dy, pos[1] + dx
            if self.env.walls[y, x]:
                y, x = pos
            scores.append(score((y, x)))
        scores = np.array(scores)
        return int(self.rng.choice(np.flatnonzero(scores == scores.max())))

    def seeker_actions(self):
        env = self.env
        actions = []
        for i in range(env.n_seekers):
            pos = tuple(env.seekers[i])
            visible = [tuple(env.hiders[j]) for j in np.where(env.seeker_sees[i])[0]]
            if visible:
                self.seeker_target[i] = min(visible, key=lambda h: self.dist[pos][h])
            elif self.seeker_target[i] == pos:
                self.seeker_target[i] = None      # reached the last known spot, nobody there
            target = self.seeker_target[i]
            if target is None:
                actions.append(int(self.rng.integers(0, 5)))
            else:
                actions.append(self._best_move(pos, lambda c: -self.dist[c][target]))
        return np.array(actions)

    def hider_actions(self):
        env = self.env
        actions = []
        for j in range(env.n_hiders):
            pos = tuple(env.hiders[j])
            if not env.hider_alive[j]:
                actions.append(0)
                continue
            visible = [tuple(env.seekers[i]) for i in np.where(env.hider_sees[j])[0]]
            if visible:
                self.hider_threat[j] = (min(visible, key=lambda s: self.dist[pos][s]), 0)
            elif self.hider_threat[j] is not None:
                spot, age = self.hider_threat[j]
                self.hider_threat[j] = (spot, age + 1) if age + 1 <= FLEE_MEMORY else None
            if self.hider_threat[j] is None:
                actions.append(0)
            else:
                threat = self.hider_threat[j][0]
                actions.append(self._best_move(pos, lambda c: self.dist[c][threat]))
        return np.array(actions)
