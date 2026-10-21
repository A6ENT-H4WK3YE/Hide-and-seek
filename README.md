# Hide and Seek MARL

FPS-style hide and seek with line of sight, trained with multi-agent reinforcement learning.

Two teams on a map with walls, in continuous space. Hiders try to stay out of sight, seekers try to find and catch them. Each agent looks in one direction and only sees what is inside its 90 degree vision cone, so walls, corners and which way you face all matter. No shooting or planting, just hiding and hunting.

This is v3.1 (v2 plus footsteps, with walk / run as a mode you switch). The earlier grid version (360 degree vision, one cell per step) is kept in git under the tag `v1-grid`.

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
- **`three_sites.txt`** (45 x 31): original FPS-style layout with three sites, a mid room, two long lanes, rotation links between sites, and cover boxes. Hiders spawn at the top like defenders, seekers at the bottom like attackers.

Check and preview any map (also confirms every walkable cell is reachable):

```
python preview_map.py maps/three_sites.txt
```

## Rules

- **Map:** loaded from a map file; every `#` is a solid 1 x 1 square. Agents are circles (radius 0.3) with a real position and a heading, and slide along walls.
- **Actions:** every step, a move (stay, forward, back, strafe left, strafe right), a turn (-90, -30, -10, -3, 0, +3, +10, +30, +90 degrees) and whether to switch between walking and running (the mode stays until switched; every round starts walking).
- **Sound:** running is full speed (1 unit per step) but enemies within 6 cells of walking distance hear you; walking is half speed and silent. Sound goes round corners, not through walls.
- **Teams:** 2 seekers v 1 hider for now (variable sizes later).
- **Hiding phase:** seekers are frozen for the first 15 steps so hiders can get into position.
- **Vision:** a 90 degree cone, no range limit, blocked by walls. Enemies are checked exactly (straight line between centres). A seeker can see a hider's back without being seen.
- **Catch:** a seeker catches a hider it can see within 0.8 units, if the hider is within 30 degrees of where the seeker looks.
- **Round end:** 100 steps (hiders win if any survive) or all hiders caught (seekers win).
- **Reward (team based, zero-sum):** seekers get a little each step a hider is in sight (more when closer) and lose a little each step none is; +10 per catch; +10 plus a bonus for time left when all are caught; -10 if hiders survive. Hiders get exactly the opposite. Full details in `formulation.md`.

## Approach

- **Algorithm:** PPO, one shared policy per team (parameter sharing), so the same network works for any team size.
- **Self-play:** both teams learn at the same time against each other.
- **Partial observability:** each agent sees only its own cone (as rays), plus a minimap with a "last seen" memory of enemies and the team's shared "searched recently" memory. A recurrent network (GRU) comes later.
- **Training is headless.** Metrics are logged every game; a GIF and a checkpoint are saved every few thousand games from a fixed evaluation map.

## Repo layout

```
config.py      all settings in one place
preview_map.py draw a map to PNG and check it is fully connected
maps/          map files and their previews
geometry.py    moving round agents among square walls, and casting rays
env.py         the game: movement, vision cones, catches, rewards, observations
render.py      draw frames, save GIFs
test_env.py    quick checks that the game rules work
random_demo.py random agents: record one GIF, then baseline win rates and speed
model.py       TeamNet: rays + enemies + minimap in, move and turn choices + value out
train.py       PPO self-play training loop
evaluate.py    play a checkpoint against random players and the bots to check it learned
bots.py        scripted chaser seeker and fleeing hider, used as fixed benchmarks
matches.py     play any mix of trained / bot / random players (used by train and evaluate)
formulation.md the exact game definition for each version
notebooks/     paste-and-run Colab versions
runs/          training outputs (ignored by git)
```

## Roadmap

1. v1: grid world, 360 degree vision (done, tag `v1-grid`).
2. v2: continuous space with a 90 degree vision cone.
3. **v3: walk vs run with sound (current).**
4. Bigger map (`three_sites.txt`) and bigger teams.
5. Continuous turning, learned memory (GRU), communication between teammates.
6. Continuous time (positions follow differential equations).
7. 3D render (the vision rays can drive a Wolfenstein-style 3D view).

## Quick start

```
pip install -r requirements.txt
python test_env.py                         # check the rules (should say: all checks passed)
python random_demo.py                      # random 2v1 on the tiny map, GIF saved in runs/
python random_demo.py maps/tiny.txt 2 2    # random 2v2
```

## Training

```
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Codespaces / no GPU (small download)
# Mac: pip install torch     AMD/NVIDIA GPU: see setup notes
python train.py smoke 20000                # 1 minute check that everything runs
python train.py                            # full run (TOTAL_STEPS in config.py)
python evaluate.py runs/<run name>/checkpoints/latest.pt
```

Each run gets its own folder in `runs/` with a `run_card.md` (every setting, game version, git commit),
`metrics.csv`, `curves.png`, recorded GIFs and checkpoints. Ctrl+C stops training early and still saves.

**Opponent pool:** plain self-play can go round in circles (hiders pick a side, seekers learn that side,
hiders switch...). With `OPPONENT_POOL = True`, half the games are played against frozen past versions
of the other team, so agents must learn strategies that work against many opponents. The pool keeps snapshots
spread evenly over the whole run (from the first random network to the latest), so the losing team also meets weaker
opponents it can still beat.

**Benchmark bots:** at every recording, the trained teams also play the scripted bots in `bots.py`, logged in
`bots.csv` and the "vs fixed benchmark bots" panel of `curves.png`. In 2v1 on the tiny map the chaser bots beat a
random hider 89% of the time and the fleeing bot 74% of the time, so watching that number fall shows hiders learning.

**What to look for:** in self-play both teams improve together, so the training win rate can hover
around the same value even when both are learning. `evaluate.py` plays each trained team against
random opponents: trained seekers should win far more than the random baseline (about 20% in 2v1),
and trained hiders should hold random seekers well below it.

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
- **Codespaces / no GPU:** `pip install torch --index-url https://download.pytorch.org/whl/cpu` (CPU-only build, much smaller). The grid environment runs fine on CPU.

## Licence

MIT
