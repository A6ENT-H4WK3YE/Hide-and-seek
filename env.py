# Hide and seek in continuous space. Agents are circles with a real position and a heading,
# walls are the '#' squares of a map file, and each agent sees a 90 degree cone blocked by walls.
# Running is fast but makes noise that enemies nearby can hear; walking is slow and silent.
# Walk / run is a mode that stays on until the agent switches it (like toggling walk in an FPS);
# every round starts walking.
#
# Usage:
#   env = HideSeekEnv()
#   obs = env.reset()
#   obs, rewards, done, info = env.step(seeker_actions, hider_actions)
#
# Actions: one (move, turn, switch) triple per agent, e.g. seeker_actions = [[1, 4, 1], [0, 6, 0]]  (see config.py)
#          switch: 0 keep my walk / run mode, 1 switch it
# obs is a dict {"seekers": [...], "hiders": [...]}, one entry per agent. Each entry is a dict:
#   "rays":    (N_RAYS, 4)            per ray, left to right across the cone: distance to what it hit,
#                                     and whether that is a wall, a teammate or an enemy
#   "enemies": (MAX_TEAM_SIZE, 4)     enemies I can see right now, nearest first (empty slots are all 0):
#                                     seen flag, direction relative to where I look (cos, sin), distance
#   "heard":   (MAX_TEAM_SIZE, 4)     enemies I can hear running right now, nearest first, same layout
#                                     (sound goes through walls, so this is the straight-line distance)
#   "vec":     (7,)                   my position, my heading (cos, sin), time, hiding phase, am I running
#   "minimap": (N_MINIMAP_CHANNELS, MINIMAP_SIZE, MINIMAP_SIZE)   the whole map, shrunk
# rewards is a dict {"seekers": array, "hiders": array}, one number per agent.
#
# Positions are (x, y) with y pointing down, 1 unit = 1 map cell (see geometry.py).

from collections import deque
import numpy as np
import config as cfg
from geometry import move_circle, cast_rays, line_clear, wrap_angle

# move id -> (forward, right) relative to where the agent looks
MOVES = {
    0: (0, 0),    # stay
    1: (1, 0),    # forward
    2: (-1, 0),   # back
    3: (0, -1),   # strafe left
    4: (0, 1),    # strafe right
}
TURNS = np.radians(cfg.TURN_ANGLES)
NO_TURN = cfg.TURN_ANGLES.index(0)

# what a ray hit
HIT_WALL, HIT_TEAMMATE, HIT_ENEMY = 0, 1, 2
RAY_FEATURES = 4       # distance, is wall, is teammate, is enemy
ENEMY_FEATURES = 4     # seen, cos, sin, distance
VEC_DIM = 7            # x, y, cos(heading), sin(heading), time, hiding phase, running

# ray directions relative to the heading, spread evenly across the cone (left edge to right edge)
RAY_OFFSETS = np.radians(np.linspace(-cfg.VISION_CONE / 2, cfg.VISION_CONE / 2, cfg.N_RAYS))
HALF_CONE = np.radians(cfg.VISION_CONE / 2)
CATCH_ANGLE = np.radians(cfg.CATCH_ANGLE)
RAY_SAMPLE = 0.5       # spacing of points along each ray, used to mark which map cells a cone covers

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


def pool_matrix(n, m):
    # (m, n) matrix that averages n map rows (or columns) down to m minimap rows.
    # Works for shrinking (big map) and for stretching (map smaller than the minimap).
    P = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        a = int(np.floor(i * n / m))
        b = min(max(int(np.floor((i + 1) * n / m)), a + 1), n)
        P[i, a:b] = 1.0 / (b - a)
    return P


def walk_distances(walls):
    # shortest walking distance (in 4-way steps between cells) from every floor cell to every cell,
    # worked out once per map: {(row, col): (H, W) array}. Used by the bots to plan routes.
    H, W = walls.shape
    out = {}
    for src in [(y, x) for y in range(H) for x in range(W) if not walls[y, x]]:
        d = np.full((H, W), 10_000)
        d[src] = 0
        queue = deque([src])
        while queue:
            y, x = queue.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and not walls[ny, nx] and d[ny, nx] == 10_000:
                    d[ny, nx] = d[y, x] + 1
                    queue.append((ny, nx))
        out[src] = d
    return out


def cell_of(pos):
    # (row, column) of the map cell a point is in
    return int(pos[1]), int(pos[0])


class HideSeekEnv:

    def __init__(self, map_file=cfg.MAP_FILE):
        self.grid, self.walls, self.hider_spawns, self.seeker_spawns = load_map(map_file)
        self.H, self.W = self.walls.shape
        self.rng = np.random.default_rng(cfg.SEED)
        # distances along a ray where we check which cell it passes over (long enough for the whole map)
        self.ray_samples = np.arange(0.0, np.hypot(self.H, self.W), RAY_SAMPLE)
        self.path_dist = walk_distances(self.walls)
        # running is heard within this straight-line distance, scaled to the map (bigger map, further sound)
        self.hearing_range = cfg.HEARING_FRACTION * max(self.H, self.W)

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

        # everyone starts in the middle of a spawn cell, looking in a random direction
        self.seekers = self._spawn(self.seeker_spawns, n_seekers)
        self.hiders = self._spawn(self.hider_spawns, n_hiders)
        self.seeker_heading = self.rng.uniform(-np.pi, np.pi, n_seekers)
        self.hider_heading = self.rng.uniform(-np.pi, np.pi, n_hiders)
        self.hider_alive = np.ones(n_hiders, dtype=bool)
        self.seeker_noisy = np.zeros(n_seekers, dtype=bool)   # made running noise this step
        self.hider_noisy = np.zeros(n_hiders, dtype=bool)
        self.seeker_running = np.zeros(n_seekers, dtype=bool) # walk / run mode: everyone starts walking
        self.hider_running = np.zeros(n_hiders, dtype=bool)
        self.t = 0
        self.done = False
        self.winner = None

        # each agent remembers where it last saw enemies (full-map grid, fades each step)
        self.seeker_memory = np.zeros((n_seekers, self.H, self.W), dtype=np.float32)
        self.hider_memory = np.zeros((n_hiders, self.H, self.W), dtype=np.float32)
        # each TEAM shares one "searched recently" map: 1 = inside someone's cone this step, fading to 0
        self.searched = {
            "seekers": np.zeros((self.H, self.W), dtype=np.float32),
            "hiders": np.zeros((self.H, self.W), dtype=np.float32),
        }
        self._update_sight()
        self._update_hearing()
        self._update_memory()
        return self._get_obs()

    def _spawn(self, cells, n):
        # different cells when possible, repeats only if the spawn area is too small
        replace = n > len(cells)
        idx = self.rng.choice(len(cells), size=n, replace=replace)
        return np.array([[cells[i][1] + 0.5, cells[i][0] + 0.5] for i in idx], dtype=float)

    # ---------- one step ----------
    def step(self, seeker_actions, hider_actions):
        assert not self.done, "round is over, call reset()"
        hiding_phase = self.t < cfg.HIDING_PHASE
        seeker_actions = np.asarray(seeker_actions).reshape(self.n_seekers, 3)
        hider_actions = np.asarray(hider_actions).reshape(self.n_hiders, 3)

        # hiders act (caught hiders stay where they are, silently)
        self.hider_noisy[:] = False
        for j in range(self.n_hiders):
            if self.hider_alive[j]:
                self.hiders[j], self.hider_heading[j], self.hider_running[j], self.hider_noisy[j] = self._act(
                    self.hiders[j], self.hider_heading[j], self.hider_running[j], hider_actions[j])

        # seekers act, unless they are still frozen in the hiding phase (no moving, turning, switching or noise)
        self.seeker_noisy[:] = False
        if not hiding_phase:
            for i in range(self.n_seekers):
                self.seekers[i], self.seeker_heading[i], self.seeker_running[i], self.seeker_noisy[i] = self._act(
                    self.seekers[i], self.seeker_heading[i], self.seeker_running[i], seeker_actions[i])

        self.t += 1
        self._update_sight()
        self._update_hearing()
        any_seen = bool(self.hiders_seen.any())

        # rewards, written from the seekers' side (hiders get exactly minus this)
        seeker_r = 0.0
        if not hiding_phase:
            if any_seen:
                seeker_r += cfg.REWARD_SEEN * self._closeness()
            else:
                seeker_r += cfg.REWARD_UNSEEN

        # catches: close enough, looking at the hider, and nothing in between (never while frozen)
        n_caught = 0
        if not hiding_phase:
            for j in range(self.n_hiders):
                if not self.hider_alive[j]:
                    continue
                for i in range(self.n_seekers):
                    if (self.seeker_sees[i, j] and self.pair_dist[i, j] <= cfg.CATCH_DISTANCE
                            and abs(self.pair_angle[i, j]) <= CATCH_ANGLE):
                        self.hider_alive[j] = False
                        n_caught += 1
                        break
            if n_caught > 0:
                self._update_sight()   # caught hiders leave the game, so nobody sees them any more
        seeker_r += cfg.REWARD_CATCH * n_caught

        # round end, plus the win/loss reward on the final step
        if not self.hider_alive.any():
            self.done, self.winner = True, "seekers"
            time_left = (cfg.MAX_STEPS - self.t) / (cfg.MAX_STEPS - cfg.HIDING_PHASE)
            seeker_r += cfg.REWARD_WIN + cfg.REWARD_TIME_BONUS * time_left
        elif self.t >= cfg.MAX_STEPS:
            self.done, self.winner = True, "hiders"
            seeker_r += cfg.REWARD_LOSS

        self._update_memory()

        # zero-sum between teams: hiders always get exactly minus the seekers' reward
        rewards = {
            "seekers": np.full(self.n_seekers, seeker_r, dtype=np.float32),
            "hiders": np.full(self.n_hiders, -seeker_r, dtype=np.float32),
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

    def _act(self, pos, heading, running, action):
        # switch walk / run if asked, turn, then move relative to the new heading, sliding along walls.
        # Running is full speed and makes noise (only if actually moving); walking is half speed and silent.
        move, turn, switch = int(action[0]), int(action[1]), int(action[2])
        run = (not running) if switch else bool(running)
        heading = wrap_angle(heading + TURNS[turn])
        fwd, right = MOVES[move]
        speed = cfg.MOVE_SPEED * (1.0 if run else cfg.WALK_SPEED_FACTOR)
        # "right" is 90 degrees clockwise from where the agent looks
        dx = (fwd * np.cos(heading) - right * np.sin(heading)) * speed
        dy = (fwd * np.sin(heading) + right * np.cos(heading)) * speed
        x, y = move_circle(self.walls, pos[0], pos[1], dx, dy, cfg.AGENT_RADIUS, cfg.MOVE_SUBSTEPS)
        noisy = run and move != 0
        return np.array([x, y]), heading, run, noisy

    def _closeness(self):
        # 1 when a seeker touches a hider it sees, falling to 0 at CLOSENESS_RANGE; the best pair counts
        d = self.pair_dist[self.seeker_sees]
        return float(np.max(np.clip(1.0 - d / cfg.CLOSENESS_RANGE, 0.0, 1.0)))

    # ---------- vision ----------
    def _agents(self):
        # every agent in one list (seekers first): positions, headings, team (0 seekers, 1 hiders), still playing
        pos = np.concatenate([self.seekers, self.hiders])
        heading = np.concatenate([self.seeker_heading, self.hider_heading])
        team = np.array([0] * self.n_seekers + [1] * self.n_hiders)
        active = np.concatenate([np.ones(self.n_seekers, dtype=bool), self.hider_alive])
        return pos, heading, team, active

    def _update_sight(self):
        pos, heading, team, active = self._agents()
        A, K, R = len(pos), cfg.N_RAYS, cfg.AGENT_RADIUS

        # 1. rays: every ray of every agent at once, each stopping at the first wall
        self.ray_angles = heading[:, None] + RAY_OFFSETS[None, :]                     # (A, K)
        self.wall_dist = cast_rays(self.walls, np.repeat(pos[:, 0], K), np.repeat(pos[:, 1], K),
                                   self.ray_angles.ravel()).reshape(A, K)

        # 2. a ray stops earlier if it hits another agent's body first
        dirx, diry = np.cos(self.ray_angles), np.sin(self.ray_angles)
        rel = pos[None, :, :] - pos[:, None, :]                                          # (A, B, 2): viewer a -> agent b
        along = rel[:, None, :, 0] * dirx[:, :, None] + rel[:, None, :, 1] * diry[:, :, None]   # (A, K, B)
        perp2 = (rel ** 2).sum(-1)[:, None, :] - along ** 2                              # squared distance from the ray
        hits = (perp2 <= R * R) & (along > 0) & active[None, None, :] & ~np.eye(A, dtype=bool)[:, None, :]
        hit_t = np.where(hits, np.maximum(along - np.sqrt(np.maximum(R * R - perp2, 0.0)), 0.0), np.inf)
        body_t = hit_t.min(axis=2)
        nearest = hit_t.argmin(axis=2)
        self.ray_dist = np.minimum(self.wall_dist, body_t)
        same_team = team[nearest] == team[:, None]
        self.ray_hit = np.where(body_t < self.wall_dist, np.where(same_team, HIT_TEAMMATE, HIT_ENEMY), HIT_WALL)

        # 3. exact enemy sight: a straight line between centres, clear of walls, inside the viewer's cone.
        #    Not two-way any more: a seeker can see a hider's back while the hider cannot see it.
        rel = self.hiders[None, :, :] - self.seekers[:, None, :]                         # (seekers, hiders, 2)
        self.pair_dist = np.hypot(rel[..., 0], rel[..., 1])
        direction = np.arctan2(rel[..., 1], rel[..., 0])                                 # seeker -> hider
        clear = line_clear(self.walls, self.seekers[:, None, 0], self.seekers[:, None, 1],
                           self.hiders[None, :, 0], self.hiders[None, :, 1])
        self.pair_angle = wrap_angle(direction - self.seeker_heading[:, None])        # where the hider is, seen by the seeker
        hider_angle = wrap_angle(direction + np.pi - self.hider_heading[None, :])     # where the seeker is, seen by the hider
        alive = self.hider_alive[None, :]
        self.seeker_sees = clear & alive & (np.abs(self.pair_angle) <= HALF_CONE)
        self.hider_sees = (clear & alive & (np.abs(hider_angle) <= HALF_CONE)).T
        self.hiders_seen = self.seeker_sees.any(axis=0)

        # 4. which map cells each cone covers (points along every ray up to the wall, plus the last one before it)
        s = self.ray_samples
        keep = s[None, None, :] < self.wall_dist[:, :, None]                            # (A, K, S)
        px = np.concatenate([pos[:, 0, None, None] + dirx[:, :, None] * s, (pos[:, 0, None] + dirx * (self.wall_dist - 1e-6))[:, :, None]], axis=2)
        py = np.concatenate([pos[:, 1, None, None] + diry[:, :, None] * s, (pos[:, 1, None] + diry * (self.wall_dist - 1e-6))[:, :, None]], axis=2)
        keep = np.concatenate([keep, np.ones((A, K, 1), dtype=bool)], axis=2)
        who = np.broadcast_to(np.arange(A)[:, None, None], keep.shape)
        rows = np.clip(np.floor(py).astype(int), 0, self.H - 1)
        cols = np.clip(np.floor(px).astype(int), 0, self.W - 1)
        self.view = np.zeros((A, self.H, self.W), dtype=bool)
        self.view[who[keep], rows[keep], cols[keep]] = True

    def _update_hearing(self):
        # seeker_hears[i, j]: seeker i hears hider j running (and the other way round).
        # Sound goes through walls: only the straight-line distance counts, up to hearing_range.
        self.seeker_hears = np.zeros((self.n_seekers, self.n_hiders), dtype=bool)
        self.hider_hears = np.zeros((self.n_hiders, self.n_seekers), dtype=bool)
        self.sound_dist = np.zeros((self.n_seekers, self.n_hiders))     # straight-line distance between each pair
        for i in range(self.n_seekers):
            for j in range(self.n_hiders):
                self.sound_dist[i, j] = np.hypot(*(self.hiders[j] - self.seekers[i]))
                close = self.sound_dist[i, j] <= self.hearing_range
                self.seeker_hears[i, j] = close and self.hider_noisy[j] and self.hider_alive[j]
                self.hider_hears[j, i] = close and self.seeker_noisy[i] and self.hider_alive[j]

    def _update_memory(self):
        # once per step: fade old info, clear cells I can see are empty, mark enemies I see
        ns = self.n_seekers
        self.seeker_memory *= cfg.MEMORY_DECAY
        self.hider_memory *= cfg.MEMORY_DECAY
        for i in range(ns):
            self.seeker_memory[i][self.view[i]] = 0.0
            for j in np.where(self.seeker_sees[i] | self.seeker_hears[i])[0]:
                self.seeker_memory[(i,) + cell_of(self.hiders[j])] = 1.0
        for j in range(self.n_hiders):
            if not self.hider_alive[j]:
                continue
            self.hider_memory[j][self.view[ns + j]] = 0.0
            for i in np.where(self.hider_sees[j] | self.hider_hears[j])[0]:
                self.hider_memory[(j,) + cell_of(self.seekers[i])] = 1.0

        # shared team search memory: fade by 1/SEARCH_MEMORY_STEPS, then mark everything inside the team's cones
        fade = 1.0 / cfg.SEARCH_MEMORY_STEPS
        for team, idx, alive in [("seekers", range(ns), np.ones(ns, dtype=bool)),
                                 ("hiders", range(ns, ns + self.n_hiders), self.hider_alive)]:
            searched = self.searched[team]
            np.maximum(searched - fade, 0.0, out=searched)
            searched[searched < 1e-6] = 0.0   # snap rounding leftovers to exactly 0
            for k, a in enumerate(idx):
                if alive[k]:
                    searched[self.view[a]] = 1.0

    # ---------- observations ----------
    def _mark(self, layer, pos, value=1.0):
        # set every minimap cell that covers the map cell pos is in (keeps the strongest value)
        y, x = cell_of(pos)
        rows, cols = self.block_rows[y], self.block_cols[x]
        layer[rows, cols] = np.maximum(layer[rows, cols], value)

    def _slots(self, me, heading, positions, distances):
        # up to MAX_TEAM_SIZE enemies, nearest first: flag, direction relative to where I look (cos, sin), distance
        out = np.zeros((cfg.MAX_TEAM_SIZE, ENEMY_FEATURES), dtype=np.float32)
        if len(positions) > 0:
            rel = np.array(positions) - me
            angle = np.arctan2(rel[:, 1], rel[:, 0]) - heading
            for k, e in enumerate(np.argsort(distances)[:cfg.MAX_TEAM_SIZE]):
                out[k] = [1.0, np.cos(angle[e]), np.sin(angle[e]), distances[e] / cfg.RAY_NORM]
        return out

    def _agent_obs(self, a, me, heading, running, teammates, enemies_seen, enemies_heard, memory, searched_small):
        # rays: distance, plus one of wall / teammate / enemy
        rays = np.zeros((cfg.N_RAYS, RAY_FEATURES), dtype=np.float32)
        rays[:, 0] = self.ray_dist[a] / cfg.RAY_NORM
        rays[np.arange(cfg.N_RAYS), 1 + self.ray_hit[a]] = 1.0

        # enemies I can see right now (straight-line distance), and enemies I can hear (walking distance)
        enemies = self._slots(me, heading, [p for p, _ in enemies_seen], [np.hypot(*(p - me)) for p, _ in enemies_seen])
        heard = self._slots(me, heading, [p for p, _ in enemies_heard], [d for _, d in enemies_heard])

        # extra numbers: where am I, which way am I looking, how far through the round, still hiding phase, am I running?
        vec = np.array([
            me[0] / self.W,
            me[1] / self.H,
            np.cos(heading),
            np.sin(heading),
            self.t / cfg.MAX_STEPS,
            float(self.t < cfg.HIDING_PHASE),
            float(running),
        ], dtype=np.float32)

        M = cfg.MINIMAP_SIZE
        mm = np.zeros((N_MINIMAP_CHANNELS, M, M), dtype=np.float32)
        mm[MM_WALLS] = self.mm_walls
        self._mark(mm[MM_ME], me)
        for p in teammates:
            self._mark(mm[MM_TEAMMATES], p)
        mm[MM_SEARCHED] = searched_small
        for (y, x) in np.argwhere(memory > 0.01):
            self._mark(mm[MM_MEMORY], (x + 0.5, y + 0.5), memory[y, x])
        return {"rays": rays, "enemies": enemies, "heard": heard, "vec": vec, "minimap": mm}

    def _get_obs(self):
        # the shrunk team search maps are the same for every teammate, so work them out once
        small = {team: self.Ry @ self.searched[team] @ self.Rx for team in ["seekers", "hiders"]}
        ns = self.n_seekers
        seekers_obs = []
        for i in range(ns):
            mates = [self.seekers[k] for k in range(ns) if k != i]
            seen = [(self.hiders[j], None) for j in np.where(self.seeker_sees[i])[0]]
            heard = [(self.hiders[j], self.sound_dist[i, j]) for j in np.where(self.seeker_hears[i] & self.hider_alive)[0]]
            seekers_obs.append(self._agent_obs(i, self.seekers[i], self.seeker_heading[i], self.seeker_running[i], mates, seen, heard,
                                               self.seeker_memory[i], small["seekers"]))
        hiders_obs = []
        for j in range(self.n_hiders):
            mates = [p for k, p in enumerate(self.hiders) if k != j and self.hider_alive[k]]
            seen = [(self.seekers[i], None) for i in np.where(self.hider_sees[j])[0]]
            heard = [(self.seekers[i], self.sound_dist[i, j]) for i in np.where(self.hider_hears[j])[0]]
            hiders_obs.append(self._agent_obs(ns + j, self.hiders[j], self.hider_heading[j], self.hider_running[j], mates, seen, heard,
                                              self.hider_memory[j], small["hiders"]))
        return {"seekers": seekers_obs, "hiders": hiders_obs}

    # ---------- helpers for rendering and bots ----------
    def cone_polygon(self, a):
        # outline of agent a's cone (a is an index into seekers then hiders): its position and the ray ends at the walls
        pos = np.concatenate([self.seekers, self.hiders])[a]
        ends = pos[None, :] + self.wall_dist[a][:, None] * np.stack([np.cos(self.ray_angles[a]), np.sin(self.ray_angles[a])], axis=1)
        return np.vstack([pos[None, :], ends])
