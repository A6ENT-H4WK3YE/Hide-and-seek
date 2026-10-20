# All settings in one place. Change numbers here, not inside the other files.

# ---------- map ----------
# Maps are text files in maps/:  '#' wall, '.' floor, 'h' hider spawn cell,
# 's' seeker spawn cell, 'A' 'B' 'C' site cells (every non-'#' cell is walkable)
# Stage 1: maps/tiny.txt (11x11, fast debugging)
# Stage 2: maps/three_sites.txt (45x31, Valorant-style) -> raise MAX_STEPS to ~250,
#          VISION_RANGE to ~7 and HIDING_PHASE to ~30 when you switch
MAP_FILE = "maps/tiny.txt"
FIXED_EVAL_SEED = 0     # same spawns for every recorded GIF, for fair comparison

# ---------- teams ----------
MIN_TEAM_SIZE = 1
MAX_TEAM_SIZE = 5
RANDOM_TEAM_SIZES = False   # False: always use the sizes below; True: random each round
N_SEEKERS = 1
N_HIDERS = 1

# ---------- round ----------
MAX_STEPS = 100         # time limit per round (discrete steps)
HIDING_PHASE = 15       # seekers frozen for this many steps at the start

# ---------- vision and catching ----------
VISION_RANGE = 5        # how far an agent can see (cells), walls block line of sight
OBS_WINDOW = 9          # each agent sees a OBS_WINDOW x OBS_WINDOW square centred on itself
CATCH_DISTANCE = 1      # caught if a seeker is within this many cells (1 = adjacent)
MEMORY_DECAY = 0.9      # how fast the "last seen" memory fades each step

# ---------- rewards ----------
REWARD_HIDDEN = 1.0     # hiders +1 / seekers -1 when no hider is visible
REWARD_SEEN = -1.0      # hiders -1 / seekers +1 when any hider is visible
REWARD_CATCH = 5.0      # bonus to seekers (penalty to hiders) per catch

# ---------- actions ----------
# 0 stay, 1 up, 2 down, 3 left, 4 right
N_ACTIONS = 5

# ---------- training ----------
SEED = 42
TOTAL_GAMES = 200_000
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
PPO_EPOCHS = 4
BATCH_GAMES = 64        # games collected before each PPO update
DEVICE = "cuda"         # "cuda" also works for AMD ROCm builds of PyTorch; use "cpu" if no GPU

# ---------- logging ----------
RUN_DIR = "runs"
RECORD_EVERY = 2_000    # save a GIF + checkpoint every this many games
LOG_EVERY = 100         # print metrics every this many games
