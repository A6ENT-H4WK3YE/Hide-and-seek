# All settings in one place. Change numbers here, not inside the other files.

# ---------- version ----------
# Matches a section in docs/formulation.md. Bump it whenever the game rules change
# (vision cone -> "v2", sound -> "v3", ...) so every run card says which game it trained on.
GAME_VERSION = "v1"

# ---------- map ----------
# Maps are text files in maps/:  '#' wall, '.' floor, 'h' hider spawn cell,
# 's' seeker spawn cell, 'A' 'B' 'C' site cells (every non-'#' cell is walkable)
# Stage 1: maps/tiny.txt (11x11, fast debugging)
# Stage 2: maps/three_sites.txt (45x31, FPS-style) -> raise MAX_STEPS to ~250,
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
# All rewards are shared by the whole team, and seekers always get minus what hiders get.
REWARD_HIDDEN = 1.0     # per step: hiders +1 / seekers -1 when no hider is visible
REWARD_SEEN = -1.0      # per step: hiders -1 / seekers +1 when any hider is visible
REWARD_CATCH = 5.0      # per catch: seekers +5 / hiders -5
REWARD_WIN = 10.0       # end of round: winning team +10, losing team -10

# ---------- actions ----------
# 0 stay, 1 up, 2 down, 3 left, 4 right
N_ACTIONS = 5

# ---------- training (PPO self-play) ----------
SEED = 42
TOTAL_STEPS = 2_000_000 # total environment steps across all parallel games
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
HIDDEN = 256            # size of the network's hidden layer
DEVICE = "cpu"         # "auto" picks the GPU if there is one (NVIDIA/AMD "cuda", Mac "mps"), else CPU

# ---------- logging ----------
RUN_DIR = "runs"
RECORD_EVERY = 2_000    # save a GIF + checkpoint every this many finished games
WINRATE_WINDOW = 200    # win rate is averaged over the last this many games
