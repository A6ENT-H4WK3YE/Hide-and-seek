# Hide and Seek MARL

FPS-style hide and seek, trained with multi-agent reinforcement learning.

Two teams on a map with walls, in continuous space. Hiders try to stay out of sight, seekers try to find and catch them. Each agent looks in one direction and only sees what is inside its 90 degree vision cone, and it can hear enemies who run nearby, so walls, corners, which way you face and how much noise you make all matter. No shooting or planting, just hiding and hunting.

## The aim

Recreate as much of an FPS game (CS / Valorant style) as possible, train agents on it with multi-agent RL, and study what they learn: tactics, teamwork and emergent behaviour. The model moves step by step towards being fully continuous (continuous space now, continuous time later, possibly learned dynamics such as neural ODEs), and the end point is a 3D render of the agents playing.

## Progress

| Version | What it added | Status |
|---|---|---|
| v1 | grid world, 360 degree vision, catch by standing next to a hider | done (kept as tag `v1-grid`) |
| v1.1 / v1.2 | bigger observation window; team-shared "searched recently" memory and a minimap | done |
| v2 | continuous space, 90 degree vision cone, rays, exact line of sight, catch by distance and facing, rewards that cannot be farmed | done |
| v2.1 | stronger reward for keeping a hider in sight (seekers had learned to spin) | done |
| big map | training on `three_sites.txt` (3 sites, mid, lanes), 250-step rounds | done, now the default |
| v3 | walk (half speed, silent) or run (full speed, heard by nearby enemies) | done |
| v3.1 | walk / run as a mode you switch, starting in walk; opponent pool spanning the whole run; games run in parallel on all CPU cores | done (tag `v3.1`; run stopped at 3.4M steps) |
| **v3.2** | sound goes through walls, and its range scales with the map (about 11 cells on the big map) | **current**, trained (seekers dominate) |

Next: team spotting (teammates share what they see and hear), an aim reward (keep the target centred), then continuous turning, learned memory (GRU), communication, continuous time, and a 3D render.

## Results so far

Seekers' win rate over 300 games per matchup, 2 seekers v 1 hider. "Chaser" and "fleeing bot" are the scripted benchmark players in `bots.py`.

| Version, map | Trained seekers v random hider | Trained seekers v fleeing bot | Chaser bot v trained hiders (lower = better hiders) | Self-play |
|---|---|---|---|---|
| v2, tiny | 65% (random seekers 22%) | 21% (chaser 79%) | 91% (random hider 88%) | 41% |
| v2.1, tiny | 96% | 67% (chaser 79%) | 91% (random hider 88%) | 86% |
| v2.1, big map | 77% (random seekers 1%) | 37% (chaser 51%) | **54%** (random hider 70%, fleeing bot 51%) | 37% |
| v3, big map | 71% | **74%** (chaser 80%) | 94% (random hider 96%) | 85% |
| v3.1, big map (stopped at 3.4M steps) | 84% | 41% (chaser 80%) | **87%** (random hider 96%) | 31% |
| v3.2, big map | 99% | 76% (chaser 91%) | 99% (random hider 100%) | 98% |

Bot strength changes between versions (from v3 the bots also hear), so compare each trained number with the bot numbers in the same row.

What the agents learned:

- **Seekers found a lazy trick first.** In v2 they strafed sideways while spinning, sweeping the room like a lighthouse, and never walked at a hider they could see: the reward for keeping someone in view was too small to bother. Raising it (v2.1) made them lock on and chase.
- **Hiders only learned on the big map.** On the 11 x 11 map a searching seeker finds you anywhere. On `three_sites.txt` hiders learned to hide about as well as the scripted fleeing bot, and picked a side: they rotated to site A, while seekers pushed straight up the middle and then spread to the sites.
- **Sound helped the seekers far more than the hiders.** In v3 seekers learned to run, listen and chase (they beat the fleeing bot 74% of the time, up from 37%), and their strafing habit disappeared. Hiders kept running and gave themselves away. Forcing the same trained hiders to walk cut the seekers' win rate from 77% to 48%, so silence pays, they just never discovered it. v3.1 makes walking a mode you switch, so staying quiet takes one decision instead of one every step.
- **Teams pick a side, and it moves.** Hiders favoured site A in v2.1 and site C in v3.
- **Sound through walls helps the hunters.** In v3.2 two seekers listening for one hider won 98% of games. Seekers learned to hold their walk / run mode; hiders kept flipping it almost every step, so they made noise half the time. v3.1 (sound blocked by walls) was the most balanced game so far (31%).

Insights so far:

- **Whichever team can never win stops learning** (seen with seekers in 1v1 on the grid and with hiders on the tiny map and in v3). Balance matters as much as the algorithm.
- **Agents do exactly what the reward pays for**, including tricks like spinning; small reward changes reshape behaviour.
- **Teamwork needs information.** Shared rewards alone gave little cooperation: a seeker cannot help with a chase it does not know about.
- **Measure behaviour, not just win rates.** Move and turn counts, forced-action tests and heatmaps found every real problem; the training curves alone did not.
- **For a small network, the game is the bottleneck.** The Mac GPU was slower than the CPU; running the games in parallel worker processes made training 2.6x faster.

The full history, every rule change and why, and all experiment notes are in `formulation.md`.

## Maps

Maps are plain text files in `maps/`, so you can draw your own in any text editor.

| Symbol | Meaning |
|---|---|
| `#` | wall |
| `.` | floor |
| `h` | hider spawn cell |
| `s` | seeker spawn cell |
| `A` `B` `C` | site cells (walkable, labelled for rendering) |

Included maps:

- **`tiny.txt`** (11 x 11): small debugging map. Train here first so bugs show up in minutes, not hours.
- **`three_sites.txt`** (45 x 31): original FPS-style layout with three sites, a mid room, two long lanes, rotation links between sites, and cover boxes. Hiders spawn at the top like defenders, seekers at the bottom like attackers. This is the default map.

Check and preview any map (also confirms every walkable cell is reachable):

```
python preview_map.py maps/three_sites.txt
```

## Rules (v3.2)

- **Map:** loaded from a map file; every `#` is a solid 1 x 1 square. Agents are circles (radius 0.3) with a real position and a heading, and slide along walls.
- **Actions:** every step, a move (stay, forward, back, strafe left, strafe right), a turn (-90, -30, -10, -3, 0, +3, +10, +30, +90 degrees) and whether to switch between walking and running (the mode stays until switched; every round starts walking).
- **Sound:** running is full speed (1 unit per step) but enemies hear you within a quarter of the map's size (about 11 cells on the big map), even through walls; walking is half speed and silent.
- **Teams:** 2 seekers v 1 hider for now (variable sizes later).
- **Hiding phase:** seekers are frozen for the first 30 steps (15 on the tiny map) so hiders can get into position.
- **Vision:** a 90 degree cone, no range limit, blocked by walls. Enemies are checked exactly (straight line between centres). A seeker can see a hider's back without being seen.
- **Catch:** a seeker catches a hider it can see within 0.8 units, if the hider is within 30 degrees of where the seeker looks.
- **Round end:** 250 steps on the big map, 100 on the tiny one (hiders win if any survive), or all hiders caught (seekers win).
- **Reward (team based, zero-sum):** seekers get a little each step a hider is in sight (more when closer) and lose a little each step none is; +10 per catch; +10 plus a bonus for time left when all are caught; -10 if hiders survive. Hiders get exactly the opposite. Full details in `formulation.md`.

## Approach

- **Algorithm:** PPO, one shared policy per team (parameter sharing), so the same network works for any team size.
- **Self-play:** both teams learn at the same time against each other.
- **Opponent pool:** half the games are played against frozen past versions of the other team, spread over the whole run, so agents must beat many opponents (stops self-play going round in circles) and the losing team still meets opponents it can beat.
- **Partial observability:** each agent sees only its own cone (as rays) and hears nearby running enemies, plus a minimap with a "last seen" memory of enemies and the team's shared "searched recently" memory. A recurrent network (GRU) comes later.
- **Parallel games:** the training games run in separate worker processes (`N_WORKERS` in config), one per CPU core; the network stays on the CPU in the main process (faster than a GPU for a network this small).
- **Fixed benchmarks:** at every recording the trained teams also play scripted bots, whose strength never changes, so progress is visible even when self-play win rates hover.
- **Training is headless.** Metrics are logged every update; a GIF and a checkpoint are saved every few thousand games from a fixed evaluation game.

## Repo layout

```
config.py      all settings in one place
preview_map.py draw a map to PNG and check it is fully connected
maps/          map files and their previews
geometry.py    moving round agents among square walls, and casting rays
env.py         the game: movement, vision cones, hearing, catches, rewards, observations
render.py      draw frames, save GIFs
test_env.py    quick checks that the game rules work (56 checks)
random_demo.py random agents: record one GIF, then baseline win rates and speed
model.py       TeamNet: rays + enemies seen + enemies heard + minimap in; move, turn and walk/run choices + value out
train.py       PPO self-play training loop
workers.py     run the training games in parallel processes, so every CPU core is used
evaluate.py    play a checkpoint against random players and the bots to check it learned
bots.py        scripted chaser seeker and fleeing hider, used as fixed benchmarks
matches.py     play any mix of trained / bot / random players (used by train and evaluate)
formulation.md the exact game definition for each version, changelog and experiment notes
notebooks/     paste-and-run Colab versions
runs/          training outputs (ignored by git)
```

## Quick start

```
pip install -r requirements.txt
python test_env.py                         # check the rules (should say: all checks passed)
python random_demo.py                      # random 2v1 on the default map, GIF saved in runs/
python random_demo.py maps/tiny.txt 2 2    # random 2v2 on the tiny map
```

## Training

```
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Codespaces / no GPU (small download)
# Mac: pip install torch     AMD/NVIDIA GPU: see setup notes
python train.py smoke 20000                # quick check that everything runs
caffeinate -i python train.py              # full run (TOTAL_STEPS in config.py; about 45 minutes for 5M steps on a 15-core Mac)
python evaluate.py runs/<run name>/checkpoints/latest.pt
```

Each run gets its own folder in `runs/` with a `run_card.md` (every setting, game version, git commit),
`metrics.csv`, `curves.png`, recorded GIFs and checkpoints. Ctrl+C stops training early and still saves.

**Benchmark bots:** at every recording, the trained teams also play the scripted bots in `bots.py`, logged in
`bots.csv` and the "vs fixed benchmark bots" panel of `curves.png`. The reference numbers for the current map and
version are measured at the start of each run and written to the run card.

**What to look for:** in self-play both teams improve together, so the training win rate can hover
around the same value even when both are learning. `evaluate.py` plays each trained team against
random players and the bots: trained seekers should win far more than random seekers (about 1% on the big map),
and trained hiders should be caught by the chaser bot less often than a random hider is. Watch the GIFs too:
a dashed ring around an agent means it ran that step and could be heard.

## What each agent sees

**Rays:** 31 rays across its 90 degree cone (one every 3 degrees). Each says how far it reaches and whether it hit a
wall, a teammate or an enemy. This is the layout in front of it, like a very low resolution screen.

**Enemies in sight:** up to 5, nearest first: which direction (relative to where it looks) and how far.

**Enemies heard:** up to 5 enemies heard running, nearest first: which direction and how far (walking distance).

**Minimap:** the whole map shrunk to 12 x 12, with walls, itself, teammates, where **its team** searched recently
(inside anyone's cone in the last 10 steps, shared, fading) and its own "last seen enemy" memory.
Same size on every map, so a network trained on the tiny map can move to the big one.

**Plus 7 numbers:** its position, which way it looks, how far through the round it is, whether it's still the hiding phase, and whether it's running.

## Setup notes

PyTorch is installed separately because the right build depends on your GPU:

- **AMD (e.g. RX 9070 GRE) on Windows:** AMD's PyTorch on Windows build (ROCm). See the AMD ROCm on Radeon docs.
- **NVIDIA:** the CUDA build from pytorch.org.
- **Mac:** `pip install torch`. Training runs on the CPU (`DEVICE = "cpu"` in config): for this small network it is faster than the Mac GPU ("mps").
- **Codespaces / no GPU:** `pip install torch --index-url https://download.pytorch.org/whl/cpu` (CPU-only build, much smaller). The game itself only needs a CPU.

## Licence

MIT
