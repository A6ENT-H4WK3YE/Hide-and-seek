# All settings in one place. Change numbers here, not inside the other files.

# ---------- version ----------
# Matches a section in formulation.md. Bump it whenever the game rules change
# (vision cone -> "v2", sound -> "v3", ...) so every run card says which game it trained on.
GAME_VERSION = "v3.3"

# ---------- map ----------
# Maps are text files in maps/:  '#' wall, '.' floor, 'h' hider spawn cell,
# 's' seeker spawn cell, 'A' 'B' 'C' site cells (every non-'#' cell is walkable)
# Stage 1: maps/tiny.txt (11x11, fast debugging)
# Stage 2: maps/three_sites.txt (45x31, FPS-style) -> raise MAX_STEPS to ~250
#          and HIDING_PHASE to ~30 when you switch
# Every '#' cell becomes a solid 1 x 1 square; agents move freely (continuously) on the rest.
MAP_FILE = "maps/three_sites.txt"
FIXED_EVAL_SEED = 0     # same spawns for every recorded GIF, for fair comparison

# ---------- teams ----------
MIN_TEAM_SIZE = 1
MAX_TEAM_SIZE = 5
RANDOM_TEAM_SIZES = False   # False: always use the sizes below; True: random each round
N_SEEKERS = 2           # 1v1 on tiny.txt is always a hider win with perfect play (see formulation.md)
N_HIDERS = 1

# ---------- round ----------
MAX_STEPS = 250         # time limit per round (discrete steps); 100 on tiny.txt
HIDING_PHASE = 30       # seekers frozen for this many steps at the start; 15 on tiny.txt

# ---------- bodies and movement (1 unit = 1 map cell, 1 step = about 1 second) ----------
AGENT_RADIUS = 0.3      # agents are circles 0.6 wide, so they fit through 1-wide corridors
MOVE_SPEED = 1.0        # units per step when running, the same for forward, back and strafing
WALK_SPEED_FACTOR = 0.5 # walking is this fraction of running speed, but silent (like holding Shift in CS)
MOVE_SUBSTEPS = 4       # each move is done in this many small pieces so nobody clips through a corner
TURN_ANGLES = [-90, -30, -10, -3, 0, 3, 10, 30, 90]   # degrees per step; positive = clockwise on screen

# ---------- vision and catching ----------
VISION_CONE = 90        # degrees, centred on where the agent is looking; no range limit, walls block it
N_RAYS = 31             # rays spread evenly across the cone (one every 3 degrees) describe the layout
RAY_NORM = 20.0         # distances in observations are divided by this (keeps the numbers near 0 to 1)
CATCH_DISTANCE = 0.8    # caught if a seeker's centre is within this distance of the hider's centre...
CATCH_ANGLE = 30        # ...the hider is within this many degrees of where the seeker looks, and in sight
MEMORY_DECAY = 0.9      # how fast each agent's own "last seen enemy" memory fades each step

# ---------- sound ----------
HEARING_FRACTION = 0.25 # an enemy running (and moving) is heard within this straight-line distance, as a fraction of
                        # the map's longer side: about 11 cells on three_sites (45 x 31), about 3 on tiny (11 x 11).
                        # Sound goes through walls (v3 / v3.1 used 6 cells of walking distance, which walls blocked)

# ---------- team search memory and minimap ----------
SEARCH_MEMORY_STEPS = 10  # shared "searched recently" marks fade to 0 over this many steps
                          # (think of a step as about a second, so roughly 10 seconds)
MINIMAP_SIZE = 12         # the whole map is shrunk to MINIMAP_SIZE x MINIMAP_SIZE for every agent

# ---------- rewards ----------
# Written from the seekers' side. All rewards are shared by the whole team,
# and hiders always get exactly minus what seekers get (zero-sum).
# The per-step rewards are small, so following a hider without catching it never pays more than catching.
REWARD_SEEN = 0.2       # per step a hider is in sight: seekers + REWARD_SEEN x closeness (v2: 0.05)
CLOSENESS_RANGE = 10.0  # closeness is 1 when touching, falling evenly to 0 at this distance
REWARD_UNSEEN = -0.05   # per step no hider is in sight
REWARD_CATCH = 10.0     # per catch
REWARD_WIN = 10.0       # all hiders caught...
REWARD_TIME_BONUS = 10.0  # ...plus up to this much more for the time left (earlier = bigger)
REWARD_LOSS = -10.0     # hiders survive to the time limit
HIDER_NOISE_COST = 0.02 # hiders only: they lose this every step a hider runs and moves (makes noise). This breaks the
                        # zero-sum rule on purpose: it is a hint for the hiders while the seekers are frozen (v3.3)

# ---------- actions ----------
# every step each agent picks two things at once:
#   move: 0 stay, 1 forward, 2 back, 3 strafe left, 4 strafe right (relative to where it looks)
#   turn: an index into TURN_ANGLES
#   switch: 0 keep my walk / run mode, 1 switch it (walk = half speed, silent; run = full speed, can be heard).
#           The mode stays until switched again, and every round starts walking.
N_MOVES = 5
N_TURNS = len(TURN_ANGLES)
N_SPEEDS = 2
SWITCH_KEEP_PROB = 0.95 # an untrained network presses "keep" this often (switches 5% of the time), so its walk / run
                        # mode stays put for long stretches; with 50 / 50 the mode flipped almost every step

# ---------- training (PPO self-play) ----------
SEED = 42
TOTAL_STEPS = 5_000_000 # total environment steps across all parallel games (2M was enough on tiny.txt)
N_ENVS = 16             # games played in parallel (one network call covers all of them)
ROLLOUT_STEPS = 128     # steps per game collected before each update (16 x 128 = 2048 per update)
LEARNING_RATE = 3e-4
GAMMA = 0.99            # discount: how much future rewards count
GAE_LAMBDA = 0.95       # advantage smoothing (bias vs variance)
CLIP_EPS = 0.2          # PPO: how far one update may move the policy
PPO_EPOCHS = 4          # passes over each batch of experience
MINIBATCHES = 4         # each batch is split into this many chunks per pass
ENT_COEF = 0.01         # bonus for keeping the policy random (exploration)
VF_COEF = 0.5           # weight of the value (critic) loss
MAX_GRAD_NORM = 0.5     # gradient clipping for stability
HIDDEN = 256            # size of the network's hidden layers
DEVICE = "cpu"          # "auto" picks the GPU if there is one (NVIDIA/AMD "cuda", Mac "mps"), else CPU.
                        # CPU is faster here: the network is small and acts on only 32 agents at a time (measured:
                        # choosing actions 16x slower on the Mac GPU, learning no faster)
N_WORKERS = 16          # the games run in this many parallel processes (workers.py); one per game was fastest
                        # (6x faster game stepping than one process on a 15-core M5 Pro)

# ---------- which teams learn ----------
# Both teams learn by default (self-play). With only "hiders" here, the seekers are frozen: they play with the
# networks saved by an earlier run (FROZEN_RUN), the final one in normal games and the run's earlier checkpoints
# (from weak to strong) as the opponent pool, so the new hiders also meet seekers they can beat.
TRAIN_TEAMS = ["hiders"]
FROZEN_RUN = "runs/v3.2_three_sites_2v1_20261021_170702"

# ---------- opponent pool (stops self-play going round in circles) ----------
# Plain self-play can cycle: hiders pick a side, seekers learn that side, hiders switch, and so on.
# Playing some games against frozen PAST versions of the other team forces strategies that beat
# many opponents, not just the current one (the idea behind fictitious play).
OPPONENT_POOL = True    # False = plain self-play
POOL_PROB = 0.5         # chance a game uses a past version of one team (which team is picked at random)
SNAPSHOT_EVERY = 10     # add a snapshot of both teams to the pool every this many updates (at the start, see below)
POOL_SIZE = 20          # keep at most this many snapshots per team. When the pool is full, every second snapshot
                        # is dropped and the gap between snapshots doubles, so the pool always spans the whole run
                        # (from the first random network to the latest), evenly spaced

# ---------- benchmark bots (see bots.py) ----------
BOT_EVAL_GAMES = 100    # games per bot matchup at every recording (0 switches it off)

# ---------- logging ----------
RUN_DIR = "runs"
RECORD_EVERY = 2_000    # save a GIF + checkpoint every this many finished games
WINRATE_WINDOW = 200    # win rate is averaged over the last this many games
