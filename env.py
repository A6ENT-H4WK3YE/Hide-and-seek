# Hide and seek environment on a grid map with line of sight.
#
# Usage:
#   env = HideSeekEnv()
#   obs = env.reset()
#   obs, rewards, done, info = env.step(seeker_actions, hider_actions)
#
# obs is a dict {"seekers": [...], "hiders": [...]}, one entry per agent.
# Each entry is a dict:
#   "grid":    (N_CHANNELS, OBS_WINDOW, OBS_WINDOW)   close-up square around the agent
#   "minimap": (N_MINIMAP_CHANNELS, MINIMAP_SIZE, MINIMAP_SIZE)   the whole map, shrunk
#   "vec":     (4,)                                   position, time, hiding phase
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
CH_SEARCHED = 5    # cells my TEAM looked at recently (shared, fades to 0 over SEARCH_MEMORY_STEPS)
N_CHANNELS = 6

# minimap channels (the whole map shrunk to MINIMAP_SIZE x MINIMAP_SIZE)
MM_WALLS = 0       # fraction of each block that is wall
MM_ME = 1          # where I am
MM_TEAMMATES = 2   # where my teammates are
MM_SEARCHED = 3    # how recently my team searched each block (average over the block)
MM_MEMORY = 4      # where I last saw enemies (strongest memory in each block)
N_MINIMAP_CHANNELS = 5


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


def pool_matrix(n, m):
    # (m, n) matrix that averages n map rows (or columns) down to m minimap rows.
    # Works for shrinking (big map) and for stretching (map smaller than the minimap).
    P = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        a = int(np.floor(i * n / m))
        b = min(max(int(np.floor((i + 1) * n / m)), a + 1), n)
        P[i, a:b] = 1.0 / (b - a)
    return P


class HideSeekEnv:

    def __init__(self, map_file=cfg.MAP_FILE):
        self.grid, self.walls, self.hider_spawns, self.seeker_spawns = load_map(map_file)
        self.H, self.W = self.walls.shape
        assert cfg.OBS_WINDOW // 2 >= cfg.VISION_RANGE, (
            f"OBS_WINDOW {cfg.OBS_WINDOW} only reaches {cfg.OBS_WINDOW // 2} cells but VISION_RANGE is "
            f"{cfg.VISION_RANGE}: agents could be seen by enemies outside their own observation. "
            f"Set OBS_WINDOW to at least {2 * cfg.VISION_RANGE + 1}.")
        self.visible = compute_visibility(self.walls, cfg.VISION_RANGE)
        self.rng = np.random.default_rng(cfg.SEED)

        # one padded canvas reused for every observation (much faster than padding each time).
        # The border around the map counts as wall, so agents near the edge see walls there.
        self.r = cfg.OBS_WINDOW // 2
        r = self.r
        self.canvas = np.zeros((N_CHANNELS, self.H + 2 * r, self.W + 2 * r), dtype=np.float32)
        self.canvas[CH_WALLS] = 1.0
        self.canvas[CH_WALLS, r:r + self.H, r:r + self.W] = self.walls

        # minimap: averaging matrices (minimap = Ry @ full_map @ Rx), and which minimap
        # cells each map row / column belongs to (for marking single points like agents)
        M = cfg.MINIMAP_SIZE
        self.Ry = pool_matrix(self.H, M)
        self.Rx = pool_matrix(self.W, M).T
        # block_rows[y] / block_cols[x]: the minimap rows / columns that cover map row y / column x
        in_row, in_col = self.Ry > 0, self.Rx.T > 0
        self.block_rows = [slice(np.where(in_row[:, y])[0].min(), np.where(in_row[:, y])[0].max() + 1) for y in range(self.H)]
        self.block_cols = [slice(np.where(in_col[:, x])[0].min(), np.where(in_col[:, x])[0].max() + 1) for x in range(self.W)]
        self.mm_walls = self.Ry @ self.walls.astype(np.float32) @ self.Rx

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
        # each TEAM shares one "searched recently" map: 1 = looked at this step, fading to 0
        self.searched = {
            "seekers": np.zeros((self.H, self.W), dtype=np.float32),
            "hiders": np.zeros((self.H, self.W), dtype=np.float32),
        }
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
        # (a hider caught this step counts as seen, even though it is no longer "alive")
        any_seen = bool(self.hiders_seen.any()) or n_caught > 0
        if hiding_phase:
            hider_r = 0.0
        else:
            hider_r = cfg.REWARD_SEEN if any_seen else cfg.REWARD_HIDDEN
            hider_r -= cfg.REWARD_CATCH * n_caught

        # round end, plus the win/loss reward on the final step
        if not self.hider_alive.any():
            self.done, self.winner = True, "seekers"
            hider_r -= cfg.REWARD_WIN
        elif self.t >= cfg.MAX_STEPS:
            self.done, self.winner = True, "hiders"
            hider_r += cfg.REWARD_WIN

        # zero-sum between teams: seekers always get exactly minus the hiders' reward
        seeker_r = -hider_r
        rewards = {
            "seekers": np.full(self.n_seekers, seeker_r, dtype=np.float32),
            "hiders": np.full(self.n_hiders, hider_r, dtype=np.float32),
        }

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

        # shared team search memory: fade by 1/SEARCH_MEMORY_STEPS, then mark everything the team sees now
        fade = 1.0 / cfg.SEARCH_MEMORY_STEPS
        for team, positions, alive in [("seekers", self.seekers, np.ones(self.n_seekers, dtype=bool)),
                                       ("hiders", self.hiders, self.hider_alive)]:
            searched = self.searched[team]
            np.maximum(searched - fade, 0.0, out=searched)
            searched[searched < 1e-6] = 0.0   # snap rounding leftovers to exactly 0
            for k, pos in enumerate(positions):
                if alive[k]:
                    searched[self.visible[self._cell(pos)].reshape(self.H, self.W)] = 1.0

    # ---------- observations ----------
    def _mark(self, layer, pos, value=1.0):
        # set every minimap cell that covers map cell pos (keeps the strongest value)
        rows, cols = self.block_rows[pos[0]], self.block_cols[pos[1]]
        layer[rows, cols] = np.maximum(layer[rows, cols], value)

    def _minimap(self, team, pos, teammates, memory, searched_small):
        M = cfg.MINIMAP_SIZE
        mm = np.zeros((N_MINIMAP_CHANNELS, M, M), dtype=np.float32)
        mm[MM_WALLS] = self.mm_walls
        self._mark(mm[MM_ME], pos)
        for p in teammates:
            self._mark(mm[MM_TEAMMATES], p)
        mm[MM_SEARCHED] = searched_small
        for (y, x) in np.argwhere(memory > 0.01):
            self._mark(mm[MM_MEMORY], (y, x), memory[y, x])
        return mm

    def _agent_obs(self, team, pos, teammates, enemies_seen, memory, searched_small):
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
        inside[CH_SEARCHED] = self.searched[team]

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
        minimap = self._minimap(team, pos, teammates, memory, searched_small)
        return {"grid": grid, "minimap": minimap, "vec": vec}

    def _get_obs(self):
        # the shrunk team search maps are the same for every teammate, so work them out once
        small = {team: self.Ry @ self.searched[team] @ self.Rx for team in ["seekers", "hiders"]}
        seekers_obs = []
        for i in range(self.n_seekers):
            mates = [self.seekers[k] for k in range(self.n_seekers) if k != i]
            seen = [self.hiders[j] for j in np.where(self.seeker_sees[i])[0]]
            seekers_obs.append(self._agent_obs("seekers", self.seekers[i], mates, seen,
                                               self.seeker_memory[i], small["seekers"]))
        hiders_obs = []
        for j in range(self.n_hiders):
            mates = [p for k, p in enumerate(self.hiders) if k != j and self.hider_alive[k]]
            seen = [self.seekers[i] for i in np.where(self.hider_sees[j])[0]]
            hiders_obs.append(self._agent_obs("hiders", self.hiders[j], mates, seen,
                                              self.hider_memory[j], small["hiders"]))
        return {"seekers": seekers_obs, "hiders": hiders_obs}

    # ---------- helpers for rendering ----------
    def view_of(self, pos):
        return self.visible[self._cell(pos)].reshape(self.H, self.W)
