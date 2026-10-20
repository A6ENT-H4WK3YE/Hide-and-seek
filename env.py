# Hide and seek environment on a grid map with line of sight.
#
# Usage:
#   env = HideSeekEnv()
#   obs = env.reset()
#   obs, rewards, done, info = env.step(seeker_actions, hider_actions)
#
# obs is a dict {"seekers": [...], "hiders": [...]}, one entry per agent.
# Each entry is a dict {"grid": (C, W, W) array, "vec": (4,) array}.
# rewards is a dict {"seekers": array, "hiders": array}, one number per agent.

import numpy as np
import config as cfg

# action id -> (dy, dx)
MOVES = {
    0: (0, 0),   # stay
    1: (-1, 0),  # up
    2: (1, 0),   # down
    3: (0, -1),  # left
    4: (0, 1),   # right
}

# observation channels
CH_WALLS = 0       # walls (outside the map also counts as wall)
CH_TEAMMATES = 1   # teammates
CH_ENEMIES = 2     # enemies I can see right now
CH_MEMORY = 3      # where I last saw enemies (fades over time)
CH_VISIBLE = 4     # cells I can currently see
N_CHANNELS = 5


def load_map(path):
    rows = [r.rstrip("\n") for r in open(path) if r.strip()]
    grid = np.array([list(r) for r in rows])
    walls = grid == "#"
    hider_spawns = [tuple(c) for c in np.argwhere(grid == "h")]
    seeker_spawns = [tuple(c) for c in np.argwhere(grid == "s")]
    return grid, walls, hider_spawns, seeker_spawns


def line_cells(y0, x0, y1, x1):
    # Bresenham's line: every grid cell on the straight line between two cells
    cells = []
    dy, dx = abs(y1 - y0), abs(x1 - x0)
    sy = 1 if y1 > y0 else -1
    sx = 1 if x1 > x0 else -1
    err = dx - dy
    y, x = y0, x0
    while True:
        cells.append((y, x))
        if y == y1 and x == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


def compute_visibility(walls, vision_range):
    # visible[a, b] is True if cell a can see cell b (flat indices y * W + x).
    # Worked out once per map, so the game itself never traces lines.
    H, W = walls.shape
    visible = np.zeros((H * W, H * W), dtype=bool)
    free = [(y, x) for y in range(H) for x in range(W) if not walls[y, x]]
    for (y0, x0) in free:
        a = y0 * W + x0
        for (y1, x1) in free:
            if (y1 - y0) ** 2 + (x1 - x0) ** 2 > vision_range ** 2:
                continue
            blocked = False
            for (y, x) in line_cells(y0, x0, y1, x1)[1:-1]:
                if walls[y, x]:
                    blocked = True
                    break
            if not blocked:
                visible[a, y1 * W + x1] = True
    return visible


class HideSeekEnv:

    def __init__(self, map_file=cfg.MAP_FILE):
        self.grid, self.walls, self.hider_spawns, self.seeker_spawns = load_map(map_file)
        self.H, self.W = self.walls.shape
        self.visible = compute_visibility(self.walls, cfg.VISION_RANGE)
        self.rng = np.random.default_rng(cfg.SEED)

        # one padded canvas reused for every observation (much faster than padding each time).
        # The border around the map counts as wall, so agents near the edge see walls there.
        self.r = cfg.OBS_WINDOW // 2
        r = self.r
        self.canvas = np.zeros((N_CHANNELS, self.H + 2 * r, self.W + 2 * r), dtype=np.float32)
        self.canvas[CH_WALLS] = 1.0
        self.canvas[CH_WALLS, r:r + self.H, r:r + self.W] = self.walls

    # ---------- start a round ----------
    def reset(self, seed=None, n_seekers=None, n_hiders=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if n_seekers is None or n_hiders is None:
            if cfg.RANDOM_TEAM_SIZES:
                n_seekers = int(self.rng.integers(cfg.MIN_TEAM_SIZE, cfg.MAX_TEAM_SIZE + 1))
                n_hiders = int(self.rng.integers(cfg.MIN_TEAM_SIZE, cfg.MAX_TEAM_SIZE + 1))
            else:
                n_seekers, n_hiders = cfg.N_SEEKERS, cfg.N_HIDERS
        self.n_seekers, self.n_hiders = n_seekers, n_hiders

        self.seekers = self._spawn(self.seeker_spawns, n_seekers)
        self.hiders = self._spawn(self.hider_spawns, n_hiders)
        self.hider_alive = np.ones(n_hiders, dtype=bool)
        self.t = 0
        self.done = False
        self.winner = None

        # each agent remembers where it last saw enemies (full-map grid, fades each step)
        self.seeker_memory = np.zeros((n_seekers, self.H, self.W), dtype=np.float32)
        self.hider_memory = np.zeros((n_hiders, self.H, self.W), dtype=np.float32)
        self._update_vision()
        return self._get_obs()

    def _spawn(self, cells, n):
        # different cells when possible, repeats only if the spawn area is too small
        replace = n > len(cells)
        idx = self.rng.choice(len(cells), size=n, replace=replace)
        return np.array([cells[i] for i in idx], dtype=int)

    # ---------- one step ----------
    def step(self, seeker_actions, hider_actions):
        assert not self.done, "round is over, call reset()"
        hiding_phase = self.t < cfg.HIDING_PHASE

        # hiders move (caught hiders stay where they are)
        for i, a in enumerate(hider_actions):
            if self.hider_alive[i]:
                self.hiders[i] = self._move(self.hiders[i], a)

        # seekers move, unless they are still frozen in the hiding phase
        if not hiding_phase:
            for i, a in enumerate(seeker_actions):
                self.seekers[i] = self._move(self.seekers[i], a)

        self.t += 1

        # catches: a seeker on the same cell or right next to a hider (no catches while frozen)
        n_caught = 0
        if not hiding_phase:
            for j in range(self.n_hiders):
                if not self.hider_alive[j]:
                    continue
                for i in range(self.n_seekers):
                    dist = np.abs(self.seekers[i] - self.hiders[j]).sum()
                    if dist <= cfg.CATCH_DISTANCE:
                        self.hider_alive[j] = False
                        n_caught += 1
                        break

        self._update_vision()

        # team rewards: is any living hider visible to any seeker?
        any_seen = bool(self.hiders_seen.any())
        if hiding_phase:
            hider_r = 0.0
        else:
            hider_r = cfg.REWARD_SEEN if any_seen else cfg.REWARD_HIDDEN
            hider_r -= cfg.REWARD_CATCH * n_caught
        seeker_r = -hider_r

        rewards = {
            "seekers": np.full(self.n_seekers, seeker_r, dtype=np.float32),
            "hiders": np.full(self.n_hiders, hider_r, dtype=np.float32),
        }

        # round end
        if not self.hider_alive.any():
            self.done, self.winner = True, "seekers"
        elif self.t >= cfg.MAX_STEPS:
            self.done, self.winner = True, "hiders"

        info = {
            "t": self.t,
            "hiding_phase": hiding_phase,
            "any_seen": any_seen,
            "n_caught": n_caught,
            "hider_alive": self.hider_alive.copy(),
            "winner": self.winner,
        }
        return self._get_obs(), rewards, self.done, info

    def _move(self, pos, action):
        dy, dx = MOVES[int(action)]
        y, x = pos[0] + dy, pos[1] + dx
        if 0 <= y < self.H and 0 <= x < self.W and not self.walls[y, x]:
            return np.array([y, x])
        return pos  # bumped into a wall: stay put

    # ---------- vision ----------
    def _cell(self, pos):
        return pos[0] * self.W + pos[1]

    def _update_vision(self):
        # seeker_sees[i, j] is True if seeker i can see living hider j (and the other way round)
        self.seeker_sees = np.zeros((self.n_seekers, self.n_hiders), dtype=bool)
        for i in range(self.n_seekers):
            for j in range(self.n_hiders):
                if self.hider_alive[j]:
                    self.seeker_sees[i, j] = self.visible[self._cell(self.seekers[i]), self._cell(self.hiders[j])]
        self.hider_sees = self.seeker_sees.T.copy()  # line of sight works both ways
        self.hiders_seen = self.seeker_sees.any(axis=0)

        # update memories: fade old info, clear cells I can see are empty, mark enemies I see
        self.seeker_memory *= cfg.MEMORY_DECAY
        self.hider_memory *= cfg.MEMORY_DECAY
        for i in range(self.n_seekers):
            view = self.visible[self._cell(self.seekers[i])].reshape(self.H, self.W)
            self.seeker_memory[i][view] = 0.0
            for j in np.where(self.seeker_sees[i])[0]:
                y, x = self.hiders[j]
                self.seeker_memory[i, y, x] = 1.0
        for j in range(self.n_hiders):
            if not self.hider_alive[j]:
                continue
            view = self.visible[self._cell(self.hiders[j])].reshape(self.H, self.W)
            self.hider_memory[j][view] = 0.0
            for i in np.where(self.hider_sees[j])[0]:
                y, x = self.seekers[i]
                self.hider_memory[j, y, x] = 1.0

    # ---------- observations ----------
    def _agent_obs(self, pos, teammates, enemies_seen, memory):
        # draw this agent's layers onto the full-map part of the canvas
        r = self.r
        inside = self.canvas[:, r:r + self.H, r:r + self.W]
        inside[CH_TEAMMATES:] = 0.0
        for p in teammates:
            inside[CH_TEAMMATES, p[0], p[1]] = 1.0
        for p in enemies_seen:
            inside[CH_ENEMIES, p[0], p[1]] = 1.0
        inside[CH_MEMORY] = memory
        inside[CH_VISIBLE] = self.visible[self._cell(pos)].reshape(self.H, self.W)

        # cut out the OBS_WINDOW x OBS_WINDOW square centred on the agent
        # (the canvas is shifted by r, so the square starts at pos itself)
        y, x = pos
        grid = self.canvas[:, y:y + 2 * r + 1, x:x + 2 * r + 1].copy()

        # extra numbers: where am I on the map, how far through the round, still in hiding phase?
        vec = np.array([
            pos[0] / (self.H - 1),
            pos[1] / (self.W - 1),
            self.t / cfg.MAX_STEPS,
            float(self.t < cfg.HIDING_PHASE),
        ], dtype=np.float32)
        return {"grid": grid, "vec": vec}

    def _get_obs(self):
        alive_hiders = [self.hiders[j] for j in range(self.n_hiders) if self.hider_alive[j]]
        seekers_obs = []
        for i in range(self.n_seekers):
            mates = [self.seekers[k] for k in range(self.n_seekers) if k != i]
            seen = [self.hiders[j] for j in np.where(self.seeker_sees[i])[0]]
            seekers_obs.append(self._agent_obs(self.seekers[i], mates, seen, self.seeker_memory[i]))
        hiders_obs = []
        for j in range(self.n_hiders):
            mates = [p for k, p in enumerate(self.hiders) if k != j and self.hider_alive[k]]
            seen = [self.seekers[i] for i in np.where(self.hider_sees[j])[0]]
            hiders_obs.append(self._agent_obs(self.hiders[j], mates, seen, self.hider_memory[j]))
        return {"seekers": seekers_obs, "hiders": hiders_obs}

    # ---------- helpers for rendering ----------
    def view_of(self, pos):
        return self.visible[self._cell(pos)].reshape(self.H, self.W)
