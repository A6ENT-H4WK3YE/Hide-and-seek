# All settings in one place. Change numbers here, not inside the other files.

# ---------- version ----------
# Matches a section in formulation.md. Bump it whenever the game rules change
# (vision cone -> "v2", sound -> "v3", attack / defend -> "v4", ...) so every run card says which game it trained on.
# The hide-and-seek game (v1 to v3.5) is kept in git under the tag v3.5-hide-and-seek.
GAME_VERSION = "v4.1"

# ---------- map ----------
# Maps are text files in maps/:  '#' wall, '.' floor, 'h' defender spawn cell (top),
# 's' attacker spawn cell (bottom), 'A' 'B' 'C' site cells (every non-'#' cell is walkable)
# Every '#' cell becomes a solid 1 x 1 square; agents move freely (continuously) on the rest.
MAP_FILE = "maps/two_sites.txt"     # v4.1: two sites (three_sites.txt has three; with 2 defenders that was too many)
FIXED_EVAL_SEED = 0     # same spawns for every recorded GIF, for fair comparison

# ---------- teams ----------
# Two teams, as in Valorant / CS: attackers try to plant on a site, defenders try to stop them.
MAX_TEAM_SIZE = 5       # observations always have room for this many players per team (bigger teams later)
N_ATTACKERS = 2
N_DEFENDERS = 2

# ---------- round ----------
MAX_STEPS = 250         # time limit per round (1 step = about 1 second); defenders win if time runs out
SETUP_STEPS = 30        # attackers are frozen (and nobody can tag) for this many steps, so defenders can take positions

# ---------- bodies and movement (1 unit = 1 map cell) ----------
AGENT_RADIUS = 0.3      # agents are circles 0.6 wide, so they fit through 1-wide corridors
MOVE_SPEED = 1.0        # units per step when running, the same for forward, back and strafing
WALK_SPEED_FACTOR = 0.5 # walking is this fraction of running speed, but silent (like holding Shift)
MOVE_SUBSTEPS = 4       # each move is done in this many small pieces so nobody clips through a corner
TURN_ANGLES = [-90, -30, -10, -3, 0, 3, 10, 30, 90]   # degrees per step; positive = clockwise on screen

# ---------- vision ----------
VISION_CONE = 90        # degrees, centred on where the agent is looking; no range limit, walls block it
N_RAYS = 31             # rays spread evenly across the cone (one every 3 degrees) describe the layout
RAY_NORM = 20.0         # distances in observations are divided by this (keeps the numbers near 0 to 1)
MEMORY_DECAY = 0.97     # how fast the team's "last known enemy position" marks fade each step (about 30 steps)

# ---------- taking someone out: keep them in your sights (v4.1, Moeez's rule) ----------
# A player is OUT once an enemy has kept them in sight, within WATCH_ANGLE of where it looks (roughly "crosshair on
# them"), for WATCH_STEPS steps in a row. Any step without them in view restarts that count. Whoever starts watching
# first finishes first, so holding an angle beats running in blind, as in Valorant; if two players finish on the same
# step (for example watching each other), a coin flip decides who goes out. Each watcher counts on its own.
WATCH_STEPS = 10        # about 10 seconds
WATCH_ANGLE = 30        # degrees either side of where the watcher looks (inside the 90 degree vision cone)

# ---------- objective (Valorant's spike) ----------
# An attacker standing still on a site cell plants; PLANT_STEPS steps in a row (moving, being out or the setup phase
# interrupt it) drops the spike there. From then on the round timer stops and the spike counts down SPIKE_STEPS:
# attackers win if it runs out, defenders win if one of them defuses it (stands still within DEFUSE_RANGE of the
# spike for DEFUSE_STEPS in a row). Taking out every attacker after the plant does not win on its own: the spike
# still has to be defused (as in Valorant). Planting and defusing make noise. Everyone knows where a planted spike is.
PLANT_STEPS = 5
SPIKE_STEPS = 50        # long enough for defenders on the other site to run over and retake (bots: 40 -> attackers
                        # 66%, 50 -> 57% on two_sites)
DEFUSE_STEPS = 7
DEFUSE_RANGE = 1.0

# ---------- sound ----------
HEARING_FRACTION = 0.25 # running (or planting) is heard by enemies within this straight-line distance, as a fraction of
                        # the map's longer side: about 11 cells on three_sites (45 x 31). Sound goes through walls

# ---------- team information and minimap ----------
SEARCH_MEMORY_STEPS = 60  # shared "searched recently" marks fade to 0 over this many steps
MINIMAP_SIZE = 12         # the whole map is shrunk to MINIMAP_SIZE x MINIMAP_SIZE for every agent

# ---------- rewards ----------
# Written from the attackers' side; defenders always get exactly minus this (zero-sum), shared by the whole team.
# Only real game events count (takedowns, the plant, the round), plus one shaping term that provably cannot change the
# best strategy (see below).
REWARD_OUT = 10.0       # each enemy taken out
REWARD_PLANT = 5.0      # the spike goes down (a real milestone of the round, like a tag)
REWARD_WIN = 20.0       # round won (spike explodes or every defender out; for defenders: defuse, every attacker out
                        # before the plant, or time runs out before the plant)
WALL_COST = 0.02        # a player that tries to move but is mostly blocked by a wall (moves less than half the way) pays
                        # this, alone: the other team does not gain it, so it is the one part that is not zero-sum.
                        # Teaches clean movement; far too small to matter next to a round (20)
SITE_SHAPING = 1.0      # potential-based shaping for attackers: potential = -SITE_SHAPING x (walking distance from the
                        # closest attacker to a site / longest such distance on the map). The reward is
                        # GAMMA x new potential - old potential: going towards a site pays a little, walking away pays
                        # it back. This kind of shaping provably leaves the best strategies unchanged; it only helps
                        # attackers find the sites early in training. Defenders get minus it, like everything else
                        # (the potential is 0 once the spike is down)

# ---------- actions ----------
# every step each agent picks three things at once:
#   move: 0 stay, 1 forward, 2 back, 3 strafe left, 4 strafe right (relative to where it looks)
#   turn: an index into TURN_ANGLES
#   switch: 0 keep my walk / run mode, 1 switch it (walk = half speed, silent; run = full speed, can be heard).
#           The mode stays until switched again, and every round starts walking.
N_MOVES = 5
N_TURNS = len(TURN_ANGLES)
N_SPEEDS = 2
SWITCH_KEEP_PROB = 0.95 # an untrained network presses "keep" this often (switches 5% of the time)

# ---------- training (PPO self-play) ----------
SEED = 42
TOTAL_STEPS = 30_000_000  # total environment steps across all parallel games (a long run, from scratch)
N_ENVS = 16             # games played in parallel (one network call covers all of them)
ROLLOUT_STEPS = 128     # steps per game collected before each update (16 x 128 = 2048 per update)
LEARNING_RATE = 3e-4
GAMMA = 0.997           # discount: with 250-step rounds, a win at the end still counts about half at the start
GAE_LAMBDA = 0.95       # advantage smoothing (bias vs variance)
CLIP_EPS = 0.2          # PPO: how far one update may move the policy
PPO_EPOCHS = 4          # passes over each batch of experience
MINIBATCHES = 4         # each batch is split into this many chunks per pass
SEQ_LEN = 32            # teams with memory learn from pieces of this many steps (each piece starts from the exact
                        # memory it had during play); shorter than the 128-step rollout, so learning is 1.7x faster
ENT_COEF = 0.01         # bonus for keeping the policy random (exploration)
VF_COEF = 0.5           # weight of the value (critic) loss
MAX_GRAD_NORM = 0.5     # gradient clipping for stability
HIDDEN = 256            # size of the network's hidden layers
DEVICE = "cpu"          # CPU is faster than the Mac GPU for this small network (measured)
N_WORKERS = 16          # the games run in this many parallel processes (workers.py), one per game

# ---------- which teams learn ----------
TRAIN_TEAMS = ["attackers", "defenders"]
FROZEN_RUN = ""         # for a team not in TRAIN_TEAMS: the run whose networks it plays with
START_FROM = {}         # {team: checkpoint path} to start a team from a saved network ({} = everyone from scratch)
MEMORY_TEAMS = ["attackers", "defenders"]   # both teams get a learned memory (GRU, see model.py)

# ---------- opponent pool (stops self-play going round in circles) ----------
OPPONENT_POOL = True    # False = plain self-play
POOL_PROB = 0.5         # chance a game uses a past version of one team (which team is picked at random)
SNAPSHOT_EVERY = 10     # add a snapshot of both teams to the pool every this many updates (at the start)
POOL_SIZE = 20          # when full, every second snapshot is dropped and the gap doubles: the pool spans the whole run
BOT_TEAM_PROB = 0.2     # share of games where one team (picked at random) is played by the scripted bots (bots.py);
                        # only the other team learns from those games, so nobody can win by exploiting one opponent

# ---------- benchmark bots (see bots.py) ----------
BOT_EVAL_GAMES = 100    # games per bot matchup at every recording (0 switches it off)

# ---------- logging ----------
RUN_DIR = "runs"
RECORD_EVERY = 5_000    # save a GIF + checkpoint every this many finished games (each costs about half a minute)
WINRATE_WINDOW = 200    # win rate is averaged over the last this many games
