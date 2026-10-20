# Hide and Seek MARL

FPS-style hide and seek with line of sight, trained with multi-agent reinforcement learning.

Two teams on a grid map with walls. Hiders try to stay out of sight, seekers try to find and catch them. Each agent only sees what is in its line of sight, so walls and corners matter. No shooting or planting, just hiding and hunting.

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

- **Map:** grid with walls loaded from a map file. Agents move up, down, left, right, or stay.
- **Teams:** variable size, from 1v1 up to 5v5.
- **Hiding phase:** seekers are frozen for the first few steps so hiders can get into position.
- **Vision:** an agent sees a cell within its vision range only if no wall blocks the straight line to it.
- **Catch:** a hider is caught when a seeker reaches an adjacent cell (or the same cell). Agents can share cells.
- **Round end:** time limit reached (hiders win if any survive) or all hiders caught (seekers win).
- **Reward (team based, zero-sum):** each step, hiders get +1 if no hider is visible to any seeker and -1 otherwise; -5 per catch; +10 for surviving the round or -10 if all are caught. Seekers always get exactly the opposite. Full details in `docs/formulation.md`.

## Approach

- **Algorithm:** PPO, one shared policy per team (parameter sharing), so the same network works for any team size.
- **Self-play:** both teams learn at the same time against each other.
- **Partial observability:** each agent sees a local window around itself, plus a "last seen" memory layer showing where enemies were last spotted. A recurrent network (GRU) comes later.
- **Training is headless.** Metrics are logged every game; a GIF and a checkpoint are saved every few thousand games from a fixed evaluation map.

## Repo layout

```
config.py      all settings in one place
preview_map.py draw a map to PNG and check it is fully connected
maps/          map files and their previews
env.py         map, movement, line of sight, catches, rewards, observations
render.py      draw frames, save GIFs
test_env.py    quick checks that the game rules work
random_demo.py random agents: record one GIF, then baseline win rates and speed
model.py       small CNN policy and value network
train.py       PPO self-play training loop
evaluate.py    play a checkpoint against random opponents to check it learned
docs/          formulation.md: the exact game definition for each version
notebooks/     paste-and-run Colab versions
runs/          training outputs (ignored by git)
```

## Roadmap

1. Environment, random agents, rendering. Check that line of sight behaves.
2. 1v1 training on `tiny.txt`.
3. 2v2 with hiding phase. Look for teamwork.
4. Switch to `three_sites.txt`, random team sizes up to 5v5.
5. GRU memory instead of the hand-made memory layer, then communication signals.
6. Continuous-time upgrade: pursuit-evasion with momentum and turning limits.

## Quick start

```
pip install -r requirements.txt
python test_env.py                         # check the rules (should say: all checks passed)
python random_demo.py                      # random 1v1 on the tiny map, GIF saved in runs/
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

**What to look for:** in self-play both teams improve together, so the training win rate can hover
around the same value even when both are learning. `evaluate.py` plays each trained team against
random opponents: trained seekers should win far more than the random baseline (about 29% in 1v1),
and trained hiders should hold random seekers well below it.

## What each agent sees

A 9 x 9 square centred on itself, with 5 layers:

1. walls
2. teammates
3. enemies it can see right now
4. memory: where it last saw enemies, fading over time (cleared when it looks and they're gone)
5. the cells it can currently see

Plus 4 numbers: its position on the map, how far through the round it is, and whether it's still the hiding phase.

## Setup notes

PyTorch is installed separately because the right build depends on your GPU:

- **AMD (e.g. RX 9070 GRE) on Windows:** AMD's PyTorch on Windows build (ROCm). See the AMD ROCm on Radeon docs.
- **NVIDIA:** the CUDA build from pytorch.org.
- **Mac:** `pip install torch` (uses the Mac GPU, "mps", automatically).
- **Codespaces / no GPU:** `pip install torch --index-url https://download.pytorch.org/whl/cpu` (CPU-only build, much smaller). The grid environment runs fine on CPU.

## Licence

MIT
