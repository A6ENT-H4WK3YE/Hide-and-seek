# v4: a two-team attack / defend game in continuous space, in the style of Valorant / CS (without guns).
#
#   attackers (spawn at the bottom) plant the spike on a site (stand still on a site cell for PLANT_STEPS steps in a
#             row); they win if it is not defused within SPIKE_STEPS, or if every defender is out
#   defenders (spawn at the top, next to the sites) win by defusing the spike (stand still next to it for
#             DEFUSE_STEPS in a row), by taking every attacker out before the plant, or if time runs out before it
#
# Taking someone out (v4.1): keep them in sight, within WATCH_ANGLE of where you look, for WATCH_STEPS steps in a
# row. Whoever starts watching first finishes first; same-step finishes (like two players watching each other) are
# settled by a coin flip.
#
# Usage:
#   env = GameEnv()
#   obs = env.reset()
#   obs, rewards, done, info = env.step({"attackers": actions, "defenders": actions})
#
# Actions: one (move, turn, switch) triple per player (see config.py).
# obs[team] is a list with one dict per player (players who are out still get one; their steps do not count):
#   "rays":    (N_RAYS, 4)              per ray, left to right across my cone: distance, and wall / teammate / enemy
#   "enemies": (MAX_TEAM_SIZE, 9)       enemies my team knows about right now, nearest first (empty slots are 0):
#                                       known, direction from me relative to where I look (cos, sin), distance,
#                                       I see it, a teammate sees it, only heard (nobody sees it),
#                                       how far my team is towards taking it out (best watch count), planting
#   "mates":   (MAX_TEAM_SIZE - 1, 7)   my teammates: still in, direction (cos, sin), distance, how far it is
#                                       towards taking an enemy out (its best watch count), planting, sees an enemy
#   "vec":     (20,)                    my position, heading (cos, sin), time, setup phase, running, my best watch count,
#                                       my plant progress, the attackers' best plant progress (everyone hears a plant),
#                                       standing on a site, attacker or not, teammates still in, enemies still in,
#                                       spike down, spike time left, spike direction (cos, sin) and distance,
#                                       the defenders' best defuse progress
#   "minimap": (6, MINIMAP_SIZE, MINIMAP_SIZE)   walls, sites, me, teammates, where my team searched recently,
#                                       where my team last saw or heard enemies (team callouts, fading)
# rewards[team]: one number per player, shared by the team (defenders get minus the attackers: zero-sum), except a
# small cost each player pays alone for walking into a wall (WALL_COST).
#
# Positions are (x, y) with y pointing down, 1 unit = 1 map cell (see geometry.py).

from collections import deque
import numpy as np
import config as cfg
from geometry import move_circle, cast_rays, line_clear, wrap_angle

TEAMS = ["attackers", "defenders"]
OTHER = {"attackers": "defenders", "defenders": "attackers"}

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
ENEMY_FEATURES = 9
MATE_FEATURES = 7
VEC_DIM = 20

# ray directions relative to the heading, spread evenly across the cone (left edge to right edge)
RAY_OFFSETS = np.radians(np.linspace(-cfg.VISION_CONE / 2, cfg.VISION_CONE / 2, cfg.N_RAYS))
HALF_CONE = np.radians(cfg.VISION_CONE / 2)
WATCH_ANGLE = np.radians(cfg.WATCH_ANGLE)
RAY_SAMPLE = 0.5       # spacing of points along each ray, used to mark which map cells a cone covers

# minimap channels (the whole map shrunk to MINIMAP_SIZE x MINIMAP_SIZE)
MM_WALLS = 0       # fraction of each block that is wall
MM_SITES = 1       # fraction of each block that is a site
MM_ME = 2          # where I am
MM_TEAMMATES = 3   # where my teammates are
MM_SEARCHED = 4    # how recently my team searched each block (average over the block)
MM_ENEMIES = 5     # where my team last saw or heard enemies (strongest mark in each block, fading)
N_MINIMAP_CHANNELS = 6


def load_map(path):
    rows = [r.rstrip("\n") for r in open(path) if r.strip()]
    grid = np.array([list(r) for r in rows])
    walls = grid == "#"
    spawns = {
        "attackers": [tuple(c) for c in np.argwhere(grid == "s")],
        "defenders": [tuple(c) for c in np.argwhere(grid == "h")],
    }
    sites = np.isin(grid, ["A", "B", "C"])
    return grid, walls, spawns, sites


def pool_matrix(n, m):
    # (m, n) matrix that averages n map rows (or columns) down to m minimap rows.
    # Works for shrinking (big map) and for stretching (map smaller than the minimap).
    P = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        a = int(np.floor(i * n / m))
        b = min(max(int(np.floor((i + 1) * n / m)), a + 1), n)
        P[i, a:b] = 1.0 / (b - a)
    return P


def bfs(walls, sources):
    # walking distance (4-way steps between cells) from the nearest of the source cells to every cell
    H, W = walls.shape
    d = np.full((H, W), 10_000)
    queue = deque()
    for src in sources:
        d[src] = 0
        queue.append(src)
    while queue:
        y, x = queue.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not walls[ny, nx] and d[ny, nx] == 10_000:
                d[ny, nx] = d[y, x] + 1
                queue.append((ny, nx))
    return d


def walk_distances(walls):
    # walking distance from every floor cell to every cell, worked out once per map: {(row, col): (H, W) array}.
    # Used by the bots to plan routes.
    H, W = walls.shape
    return {(y, x): bfs(walls, [(y, x)]) for y in range(H) for x in range(W) if not walls[y, x]}


def cell_of(pos):
    # (row, column) of the map cell a point is in
    return int(pos[1]), int(pos[0])


class GameEnv:

    def __init__(self, map_file=cfg.MAP_FILE):
        self.grid, self.walls, self.spawns, self.sites = load_map(map_file)
        self.H, self.W = self.walls.shape
        self.rng = np.random.default_rng(cfg.SEED)
        self.ray_samples = np.arange(0.0, np.hypot(self.H, self.W), RAY_SAMPLE)
        self.path_dist = walk_distances(self.walls)
        # running and planting are heard within this straight-line distance, scaled to the map
        self.hearing_range = cfg.HEARING_FRACTION * max(self.H, self.W)
        # walking distance from every cell to the nearest site cell (for the attackers' shaping)
        self.site_dist = bfs(self.walls, [tuple(c) for c in np.argwhere(self.sites)])
        self.site_norm = float(max(self.site_dist[~self.walls].max(), 1))

        # minimap: averaging matrices (minimap = Ry @ full_map @ Rx), and which minimap
        # cells each map row / column belongs to (for marking single points like agents)
        M = cfg.MINIMAP_SIZE
        self.Ry = pool_matrix(self.H, M)
        self.Rx = pool_matrix(self.W, M).T
        in_row, in_col = self.Ry > 0, self.Rx.T > 0
        self.block_rows = [slice(np.where(in_row[:, y])[0].min(), np.where(in_row[:, y])[0].max() + 1) for y in range(self.H)]
        self.block_cols = [slice(np.where(in_col[:, x])[0].min(), np.where(in_col[:, x])[0].max() + 1) for x in range(self.W)]
        self.mm_walls = self.Ry @ self.walls.astype(np.float32) @ self.Rx
        self.mm_sites = self.Ry @ self.sites.astype(np.float32) @ self.Rx

    # ---------- start a round ----------
    def reset(self, seed=None, sizes=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if sizes is None:
            sizes = {"attackers": cfg.N_ATTACKERS, "defenders": cfg.N_DEFENDERS}
        self.n = dict(sizes)
        self.offset = {"attackers": 0, "defenders": self.n["attackers"]}   # index of each team in "all players" order

        # everyone starts in the middle of a spawn cell, looking in a random direction, walking
        self.pos = {team: self._spawn(self.spawns[team], self.n[team]) for team in TEAMS}
        self.heading = {team: self.rng.uniform(-np.pi, np.pi, self.n[team]) for team in TEAMS}
        self.alive = {team: np.ones(self.n[team], dtype=bool) for team in TEAMS}
        self.running = {team: np.zeros(self.n[team], dtype=bool) for team in TEAMS}
        self.noisy = {team: np.zeros(self.n[team], dtype=bool) for team in TEAMS}
        self.bumped = {team: np.zeros(self.n[team], dtype=bool) for team in TEAMS}     # walked into a wall this step
        self.plant = {team: np.zeros(self.n[team], dtype=int) for team in TEAMS}      # steps planted in a row
        # watch[team][i, j]: how many steps in a row my player i has kept enemy j in its sights
        self.watch = {team: np.zeros((self.n[team], self.n[OTHER[team]]), dtype=int) for team in TEAMS}
        self.defuse = np.zeros(self.n["defenders"], dtype=int)    # defenders' defuse steps in a row
        self.spike = None              # where the spike is once planted
        self.spike_timer = 0           # steps left before it explodes
        self.t = 0
        self.done = False
        self.winner = None

        # team information: where we searched recently, and where we last saw or heard enemies (shared callouts)
        self.searched = {team: np.zeros((self.H, self.W), dtype=np.float32) for team in TEAMS}
        self.enemy_memory = {team: np.zeros((self.H, self.W), dtype=np.float32) for team in TEAMS}
        self._update_sight()
        self._update_hearing()
        self._update_memory()
        self.potential = self._potential()
        return self._get_obs()

    def _spawn(self, cells, n):
        # different cells when possible, repeats only if the spawn area is too small
        replace = n > len(cells)
        idx = self.rng.choice(len(cells), size=n, replace=replace)
        return np.array([[cells[i][1] + 0.5, cells[i][0] + 0.5] for i in idx], dtype=float)

    def on_site(self, pos):
        return bool(self.sites[cell_of(pos)])

    # ---------- one step ----------
    def step(self, actions):
        assert not self.done, "round is over, call reset()"
        setup = self.t < cfg.SETUP_STEPS

        # everyone acts at once. Attackers are frozen during the setup phase. Standing still on a site plants (before
        # the plant, attackers); standing still next to the spike defuses (after it, defenders). Moving resets
        # progress. Both make noise.
        for team in TEAMS:
            acts = np.asarray(actions[team]).reshape(self.n[team], 3)
            self.noisy[team][:] = False
            self.bumped[team][:] = False
            for i in range(self.n[team]):
                if not self.alive[team][i] or (team == "attackers" and setup):
                    continue
                self.pos[team][i], self.heading[team][i], self.running[team][i], self.noisy[team][i], self.bumped[team][i] = \
                    self._act(self.pos[team][i], self.heading[team][i], self.running[team][i], acts[i])
                still = int(acts[i, 0]) == 0
                if team == "attackers":
                    planting = self.spike is None and still and self.on_site(self.pos[team][i])
                    self.plant[team][i] = self.plant[team][i] + 1 if planting else 0
                    self.noisy[team][i] |= planting
                else:
                    defusing = (self.spike is not None and still
                                and np.hypot(*(self.pos[team][i] - self.spike)) <= cfg.DEFUSE_RANGE)
                    self.defuse[i] = self.defuse[i] + 1 if defusing else 0
                    self.noisy[team][i] |= defusing

        self.t += 1
        if self.spike is not None:
            self.spike_timer -= 1
        self._update_sight()
        self._update_hearing()

        # watching and eliminations (not during the setup phase)
        outs = {team: 0 for team in TEAMS}      # enemies taken out BY each team this step
        if not setup:
            # count steps in a row each player keeps each enemy in its sights; anything else restarts the count
            for team in TEAMS:
                in_sights = self.sees[team] & (np.abs(self.angle[team]) <= WATCH_ANGLE)
                self.watch[team] = np.where(in_sights, self.watch[team] + 1, 0)
            # everyone who reaches WATCH_STEPS takes their target out, in a random order: a player taken out earlier
            # in the same step does not get its own takedown (so a mutual finish is a coin flip)
            finishes = [(team, i, j) for team in TEAMS for i, j in zip(*np.where(self.watch[team] >= cfg.WATCH_STEPS))]
            for k in self.rng.permutation(len(finishes)):
                team, i, j = finishes[k]
                victims = OTHER[team]
                if self.alive[team][i] and self.alive[victims][j]:
                    self.alive[victims][j] = False
                    self.plant[victims][j] = 0
                    if victims == "defenders":
                        self.defuse[j] = 0
                    self.noisy[victims][j] = False
                    outs[team] += 1
            if outs["attackers"] or outs["defenders"]:
                for team in TEAMS:              # nobody keeps counting on (or by) players who are out
                    self.watch[team] *= self.alive[team][:, None] & self.alive[OTHER[team]][None, :]
                self._update_sight()            # players who are out leave the game, nobody sees or hears them
                self._update_hearing()

        # rewards, written from the attackers' side (defenders get exactly minus this)
        r = cfg.REWARD_OUT * (outs["attackers"] - outs["defenders"])

        # the spike goes down when an attacker finishes planting
        planted_now = False
        done_planting = np.where(self.alive["attackers"] & (self.plant["attackers"] >= cfg.PLANT_STEPS))[0]
        if self.spike is None and len(done_planting) > 0:
            self.spike = self.pos["attackers"][done_planting[0]].copy()
            self.spike_timer = cfg.SPIKE_STEPS
            self.plant["attackers"][:] = 0
            planted_now = True
            r += cfg.REWARD_PLANT

        # round end
        att_in, def_in = self.alive["attackers"].any(), self.alive["defenders"].any()
        if self.spike is not None:
            if (self.defuse >= cfg.DEFUSE_STEPS).any():
                self.winner = "defenders"       # defused
            elif not def_in or self.spike_timer <= 0:
                self.winner = "attackers"       # nobody left to defuse, or the spike exploded
        elif att_in and not def_in:
            self.winner = "attackers"
        elif def_in and not att_in:
            self.winner = "defenders"
        elif not att_in and not def_in:
            self.winner = "draw"                # both teams out on the same step
        elif self.t >= cfg.MAX_STEPS:
            self.winner = "defenders"           # time ran out without a plant
        if self.winner is not None:
            self.done = True
            r += {"attackers": cfg.REWARD_WIN, "defenders": -cfg.REWARD_WIN, "draw": 0.0}[self.winner]

        # potential-based shaping for attackers (towards a site): GAMMA x new potential - old potential
        new_potential = 0.0 if self.done else self._potential()
        r += cfg.GAMMA * new_potential - self.potential
        self.potential = new_potential

        self._update_memory()
        rewards = {
            # (the wall cost is each player's own, and is the one part that is not zero-sum)
            "attackers": (r - cfg.WALL_COST * self.bumped["attackers"]).astype(np.float32),
            "defenders": (-r - cfg.WALL_COST * self.bumped["defenders"]).astype(np.float32),
        }
        info = {
            "t": self.t,
            "setup": setup,
            "winner": self.winner,
            "planted": self.spike is not None,
            "planted_now": planted_now,
            "defused": self.winner == "defenders" and self.spike is not None,
            "outs": outs,
            "bumped": {team: int(self.bumped[team].sum()) for team in TEAMS},
            "alive": {team: self.alive[team].copy() for team in TEAMS},
            "any_seen": bool(self.sees["attackers"].any() or self.sees["defenders"].any()),
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
        # walked into a wall: tried to move but got less than half way
        bumped = move != 0 and np.hypot(x - pos[0], y - pos[1]) < 0.5 * speed
        return np.array([x, y]), heading, run, noisy, bumped

    def _potential(self):
        # attackers' shaping potential: minus the walking distance from the closest attacker still in to a site
        # (0 once the spike is down: the job it helps with is done)
        in_game = [p for p, a in zip(self.pos["attackers"], self.alive["attackers"]) if a]
        if not in_game or self.spike is not None:
            return 0.0
        d = min(self.site_dist[cell_of(p)] for p in in_game)
        return -cfg.SITE_SHAPING * d / self.site_norm

    # ---------- vision ----------
    def all_players(self):
        # every player in one list (attackers first): positions, headings, team (0 attackers, 1 defenders), still in
        pos = np.concatenate([self.pos[t] for t in TEAMS])
        heading = np.concatenate([self.heading[t] for t in TEAMS])
        team = np.array([0] * self.n["attackers"] + [1] * self.n["defenders"])
        active = np.concatenate([self.alive[t] for t in TEAMS])
        return pos, heading, team, active

    def _update_sight(self):
        pos, heading, team, active = self.all_players()
        A, K, R = len(pos), cfg.N_RAYS, cfg.AGENT_RADIUS

        # 1. rays: every ray of every player at once, each stopping at the first wall
        self.ray_angles = heading[:, None] + RAY_OFFSETS[None, :]                     # (A, K)
        self.wall_dist = cast_rays(self.walls, np.repeat(pos[:, 0], K), np.repeat(pos[:, 1], K),
                                   self.ray_angles.ravel()).reshape(A, K)

        # 2. a ray stops earlier if it hits another player's body first (players who are out have no body)
        dirx, diry = np.cos(self.ray_angles), np.sin(self.ray_angles)
        rel = pos[None, :, :] - pos[:, None, :]                                          # (A, B, 2): viewer a -> player b
        along = rel[:, None, :, 0] * dirx[:, :, None] + rel[:, None, :, 1] * diry[:, :, None]   # (A, K, B)
        perp2 = (rel ** 2).sum(-1)[:, None, :] - along ** 2
        hits = (perp2 <= R * R) & (along > 0) & active[None, None, :] & ~np.eye(A, dtype=bool)[:, None, :]
        hit_t = np.where(hits, np.maximum(along - np.sqrt(np.maximum(R * R - perp2, 0.0)), 0.0), np.inf)
        body_t = hit_t.min(axis=2)
        nearest = hit_t.argmin(axis=2)
        self.ray_dist = np.minimum(self.wall_dist, body_t)
        same_team = team[nearest] == team[:, None]
        self.ray_hit = np.where(body_t < self.wall_dist, np.where(same_team, HIT_TEAMMATE, HIT_ENEMY), HIT_WALL)

        # 3. exact enemy sight: a straight line between centres, clear of walls, inside the viewer's cone.
        #    sees[team][i, j]: my player i sees enemy j; dist / angle: distance, and where j is relative to i's heading
        a, d = self.pos["attackers"], self.pos["defenders"]
        rel = d[None, :, :] - a[:, None, :]                                              # attacker -> defender
        dist = np.hypot(rel[..., 0], rel[..., 1])
        direction = np.arctan2(rel[..., 1], rel[..., 0])
        clear = line_clear(self.walls, a[:, None, 0], a[:, None, 1], d[None, :, 0], d[None, :, 1]).reshape(dist.shape)
        both_in = self.alive["attackers"][:, None] & self.alive["defenders"][None, :]
        att_angle = wrap_angle(direction - self.heading["attackers"][:, None])
        def_angle = wrap_angle(direction + np.pi - self.heading["defenders"][None, :])
        self.dist = {"attackers": dist, "defenders": dist.T}
        self.angle = {"attackers": att_angle, "defenders": def_angle.T}
        self.sees = {"attackers": clear & both_in & (np.abs(att_angle) <= HALF_CONE),
                     "defenders": (clear & both_in & (np.abs(def_angle) <= HALF_CONE)).T}

        # 4. which map cells each cone covers (points along every ray up to the wall, plus the last one before it)
        s = self.ray_samples
        keep = s[None, None, :] < self.wall_dist[:, :, None]
        px = np.concatenate([pos[:, 0, None, None] + dirx[:, :, None] * s, (pos[:, 0, None] + dirx * (self.wall_dist - 1e-6))[:, :, None]], axis=2)
        py = np.concatenate([pos[:, 1, None, None] + diry[:, :, None] * s, (pos[:, 1, None] + diry * (self.wall_dist - 1e-6))[:, :, None]], axis=2)
        keep = np.concatenate([keep, np.ones((A, K, 1), dtype=bool)], axis=2)
        who = np.broadcast_to(np.arange(A)[:, None, None], keep.shape)
        rows = np.clip(np.floor(py).astype(int), 0, self.H - 1)
        cols = np.clip(np.floor(px).astype(int), 0, self.W - 1)
        self.view = np.zeros((A, self.H, self.W), dtype=bool)
        self.view[who[keep], rows[keep], cols[keep]] = True

    def _update_hearing(self):
        # hears[team][i, j]: my player i hears enemy j running or planting (straight-line distance, through walls)
        self.hears = {}
        for team in TEAMS:
            enemy = OTHER[team]
            self.hears[team] = ((self.dist[team] <= self.hearing_range) & self.noisy[enemy][None, :]
                                & self.alive[team][:, None] & self.alive[enemy][None, :])

    def _update_memory(self):
        # once per step, per team: fade, then mark what the team sees now (searched) and where it sees or hears
        # enemies (callouts). Cells the team can see with no enemy in them are cleared from the enemy memory.
        fade = 1.0 / cfg.SEARCH_MEMORY_STEPS
        for team in TEAMS:
            enemy = OTHER[team]
            searched, memory = self.searched[team], self.enemy_memory[team]
            np.maximum(searched - fade, 0.0, out=searched)
            searched[searched < 1e-6] = 0.0
            memory *= cfg.MEMORY_DECAY
            for i in np.where(self.alive[team])[0]:
                view = self.view[self.offset[team] + i]
                searched[view] = 1.0
                memory[view] = 0.0
            known = self.sees[team].any(axis=0) | self.hears[team].any(axis=0)
            for j in np.where(known)[0]:
                memory[cell_of(self.pos[enemy][j])] = 1.0

    # ---------- observations ----------
    def _mark(self, layer, pos, value=1.0):
        # set every minimap cell that covers the map cell pos is in (keeps the strongest value)
        y, x = cell_of(pos)
        rows, cols = self.block_rows[y], self.block_cols[x]
        layer[rows, cols] = np.maximum(layer[rows, cols], value)

    def watch_progress(self, team, i=None, target=None):
        # how far towards a takedown (0 to 1): player i's best count on any enemy, or the team's best count on target
        if target is not None:
            counts = self.watch[team][:, target]
        else:
            counts = self.watch[team][i]
        return float(counts.max()) / cfg.WATCH_STEPS if len(counts) else 0.0

    def _player_obs(self, team, i, small_searched, small_enemies):
        enemy = OTHER[team]
        me, heading = self.pos[team][i], self.heading[team][i]
        a = self.offset[team] + i

        # rays: distance, plus one of wall / teammate / enemy
        rays = np.zeros((cfg.N_RAYS, RAY_FEATURES), dtype=np.float32)
        rays[:, 0] = self.ray_dist[a] / cfg.RAY_NORM
        rays[np.arange(cfg.N_RAYS), 1 + self.ray_hit[a]] = 1.0

        # enemies my team knows about (seen or heard by anyone on the team), nearest first
        enemies = np.zeros((cfg.MAX_TEAM_SIZE, ENEMY_FEATURES), dtype=np.float32)
        seen_team = self.sees[team].any(axis=0)
        heard_team = self.hears[team].any(axis=0)
        known = [j for j in range(self.n[enemy]) if self.alive[enemy][j] and (seen_team[j] or heard_team[j])]
        known.sort(key=lambda j: self.dist[team][i, j])
        for k, j in enumerate(known[:cfg.MAX_TEAM_SIZE]):
            rel = self.pos[enemy][j] - me
            ang = np.arctan2(rel[1], rel[0]) - heading
            mates_see = bool(np.delete(self.sees[team][:, j], i).any())
            enemies[k] = [1.0, np.cos(ang), np.sin(ang), self.dist[team][i, j] / cfg.RAY_NORM,
                          float(self.sees[team][i, j]), float(mates_see), float(not seen_team[j]),
                          self.watch_progress(team, target=j), self.plant[enemy][j] / cfg.PLANT_STEPS]

        # teammates: those still in first, nearest first
        mates = np.zeros((cfg.MAX_TEAM_SIZE - 1, MATE_FEATURES), dtype=np.float32)
        others = [k for k in range(self.n[team]) if k != i]
        others.sort(key=lambda k: (not self.alive[team][k], np.hypot(*(self.pos[team][k] - me))))
        for slot, k in enumerate(others[:cfg.MAX_TEAM_SIZE - 1]):
            if not self.alive[team][k]:
                continue
            rel = self.pos[team][k] - me
            ang = np.arctan2(rel[1], rel[0]) - heading
            mates[slot] = [1.0, np.cos(ang), np.sin(ang), np.hypot(*rel) / cfg.RAY_NORM,
                           self.watch_progress(team, i=k), self.plant[team][k] / cfg.PLANT_STEPS,
                           float(self.sees[team][k].any())]

        # extra numbers
        n_mates = self.n[team] - 1
        vec = np.array([
            me[0] / self.W,
            me[1] / self.H,
            np.cos(heading),
            np.sin(heading),
            self.t / cfg.MAX_STEPS,
            float(self.t < cfg.SETUP_STEPS),
            float(self.running[team][i]),
            self.watch_progress(team, i=i),
            self.plant[team][i] / cfg.PLANT_STEPS,
            self.plant["attackers"].max() / cfg.PLANT_STEPS,
            float(self.on_site(me)),
            float(team == "attackers"),
            (self.alive[team].sum() - self.alive[team][i]) / n_mates if n_mates > 0 else 0.0,
            self.alive[enemy].sum() / self.n[enemy],
        ] + self._spike_numbers(me, heading), dtype=np.float32)

        M = cfg.MINIMAP_SIZE
        mm = np.zeros((N_MINIMAP_CHANNELS, M, M), dtype=np.float32)
        mm[MM_WALLS] = self.mm_walls
        mm[MM_SITES] = self.mm_sites
        self._mark(mm[MM_ME], me)
        for k in others:
            if self.alive[team][k]:
                self._mark(mm[MM_TEAMMATES], self.pos[team][k])
        mm[MM_SEARCHED] = small_searched
        mm[MM_ENEMIES] = small_enemies
        return {"rays": rays, "enemies": enemies, "mates": mates, "vec": vec, "minimap": mm}

    def _spike_numbers(self, me, heading):
        # what everyone knows about a planted spike: down or not, time left, direction from me, distance, defuse progress
        if self.spike is None:
            return [0.0] * 6
        rel = self.spike - me
        ang = np.arctan2(rel[1], rel[0]) - heading
        return [1.0, self.spike_timer / cfg.SPIKE_STEPS, np.cos(ang), np.sin(ang), np.hypot(*rel) / cfg.RAY_NORM,
                self.defuse.max() / cfg.DEFUSE_STEPS]

    def _get_obs(self):
        obs = {}
        for team in TEAMS:
            # the team's shrunk maps are the same for every teammate, so work them out once
            small_searched = self.Ry @ self.searched[team] @ self.Rx
            small_enemies = np.zeros((cfg.MINIMAP_SIZE, cfg.MINIMAP_SIZE), dtype=np.float32)
            for (y, x) in np.argwhere(self.enemy_memory[team] > 0.01):
                self._mark(small_enemies, (x + 0.5, y + 0.5), self.enemy_memory[team][y, x])
            obs[team] = [self._player_obs(team, i, small_searched, small_enemies) for i in range(self.n[team])]
        return obs

    # ---------- helpers for rendering ----------
    def cone_polygon(self, a):
        # outline of player a's cone (a is an index into all_players order): its position and the ray ends at the walls
        pos = self.all_players()[0][a]
        ends = pos[None, :] + self.wall_dist[a][:, None] * np.stack([np.cos(self.ray_angles[a]), np.sin(self.ray_angles[a])], axis=1)
        return np.vstack([pos[None, :], ends])
