# CLAUDE.md: Hide and Seek MARL

Read this first. It is the project memory: what we are building, what has happened so far, what comes
next, and how Moeez wants to work. Keep it up to date when decisions are made or results come in.

## Who you are working with

Moeez (Muhammad Moeez), PhD student in Data-Driven Engineering and Science at the University of Bristol.
Background in robotics and ML (PyTorch). His PhD is data-driven mathematical modelling (JAX, Julia SciML).
This is a side project, partly to learn multi-agent RL hands-on alongside a taught unit on stochastic
optimisation (MDPs, Q-learning, stochastic games, Nash equilibria, multi-agent RL, partial observability).

## How Moeez wants to work (important)

- **Confirm before coding.** Lay out the proposed change (what, why, what it affects) and wait for a clear
  go-ahead. Never code ahead of agreement. Small fixes to code you just wrote are fine.
- **Explain in plain words.** No heavy equations unless asked. Use analogies and FPS-game terms.
- **Never use em dashes** in replies or in files.
- **One change at a time.** Each experiment should change one thing, so results can be attributed.
  If several changes must go together, say so explicitly.
- **Be direct and concise.** He dislikes long question lists; when there are choices, give a recommendation.
- **He learns by understanding, not just receiving code.** Explain what each new piece does and why.
- **Check the actual files before answering** about the code or results. Build on the existing code, do not
  rewrite from scratch.
- **Test before handing over:** run `python test_env.py` and a smoke run (`python train.py smoke 20000`)
  after any change to the game or training code.

## Machine setup

- MacBook (university-managed), Python 3.9.6 in `.venv` (`source .venv/bin/activate`). **Keep code
  Python 3.9 compatible:** no `X | Y` type hints, no `match` statements.
- Training on **CPU** (`DEVICE = "cpu"` in config). MPS (Mac GPU) was slower for this small network.
  CPU smoke run: about 1,200 steps/s in 1v1, about 730 steps/s in 2v1 with the minimap.
- Long runs: `caffeinate -i python train.py` so the Mac does not sleep.
- GitHub repo: `A6ENT-H4WK3YE/Hide-and-seek`. There is also a GitHub Codespace copy (2 cores, slow);
  the Mac is the main machine now. Commit and push before switching machines.
- He also has a desktop with an AMD RX 9070 GRE (PyTorch via ROCm on Windows is possible, not set up).

## The end goal

Recreate as much of an FPS game as possible (CS / Valorant style, no shooting yet), train agents on it
with multi-agent reinforcement learning, and study what they learn to do: tactics, teamwork, emergent
behaviour. Move towards a fully continuous model (continuous space now, continuous time later, possibly
learned dynamics such as neural ODEs, which ties into his PhD). Finish with a **3D render** showing the
agents playing in a 3D world. The goal can grow later.

Roadmap:
1. v1: grid world (done, see results)
2. **v2: continuous space with cone vision (built on branch `v2-continuous`, first training run under way)**
3. Walk vs run with sound (running is fast but makes noise; sound blocked by walls)
4. Bigger map (`maps/three_sites.txt`) and bigger teams
5. Continuous turning actions, learned memory (GRU), communication between teammates
6. Continuous time (positions follow differential equations)
7. 3D render (the vision rays can drive a Wolfenstein-style 3D view)

## Repo layout (v2)

```
config.py       all settings in one place (GAME_VERSION, bodies, vision, rewards, training, pool, bots)
geometry.py     moving round agents among square walls (slide), casting rays (all at once, cell by cell)
env.py          the game: movement, vision cones, catches, rewards, observations
render.py       draws frames and GIFs
model.py        TeamNet: one network per team, shared by teammates (parameter sharing)
train.py        PPO self-play, opponent pool, run card, metrics, GIFs, checkpoints, bot benchmarks
evaluate.py     plays a checkpoint against random players and the bots
bots.py         scripted chaser seeker and fleeing hider (fixed benchmarks)
matches.py      plays any mix of trained / bot / random players
test_env.py     rule checks (must say "all checks passed")
random_demo.py  random agents: one GIF plus baseline win rates and speed
preview_map.py  draws a map file to PNG and checks it is fully connected
formulation.md  the exact game definition per version, changelog and experiment notes
maps/           tiny.txt (11 x 11, debugging), three_sites.txt (45 x 31, FPS-style, 3 sites)
runs/           training outputs, ignored by git (each run has run_card.md, metrics.csv, curves.png,
                bots.csv, gifs/, checkpoints/)
```

Map files: `#` wall, `.` floor, `h` hider spawn, `s` seeker spawn, `A` `B` `C` site cells.

Commands:
```
python test_env.py
python train.py smoke 20000          # quick check
caffeinate -i python train.py        # full run (TOTAL_STEPS in config)
python evaluate.py runs/<run>/checkpoints/latest.pt
```

## The grid version as it stands (labelled v1.2 in config, tagged `v1-grid` in git)

- Grid map, agents move one cell per step: stay / up / down / left / right.
- 360 degree vision, range 5 cells, walls block line of sight (Bresenham lines on grid cells).
- Hiding phase: seekers frozen for the first 15 steps. Round: 100 steps.
- Catch: a seeker on the same or an adjacent cell.
- Rewards, shared by the team, zero-sum (seekers always get minus the hiders):
  per step, hiders +1 if no hider is visible, -1 if any is; -5 per catch; +10 / -10 win or loss at the end.
- Observations: 11 x 11 close-up with 6 layers (walls, teammates, visible enemies, own "last seen enemy"
  memory, visible cells, team-shared "searched recently" memory that fades to 0 over 10 steps),
  a 12 x 12 minimap of the whole map (walls, me, teammates, team searched, my enemy memory),
  plus 4 numbers (position, time, hiding phase).
- Training: PPO, both teams learning at once (self-play), with an **opponent pool**: half the games are
  against frozen past versions of one team (fixes cycling, the idea behind fictitious play).
- Benchmarks: at every recording, trained teams play the scripted bots; reference numbers are measured at
  the start of each run and written to the run card.
- Setup: 2 seekers v 1 hider.

## Results so far

**1v1, tiny map, v1 (plain self-play, 2M steps, `runs/v1_tiny_1v1_*`):**
- Hiders learned to run: random seekers won only 1.2% against trained hiders (random vs random: 33%).
- Seekers learned some searching: 50% against random hiders.
- Trained vs trained: 0% for seekers. Once hiders were good, seekers could never win, every action gave the
  same outcome, so seekers stopped searching ("learned helplessness").
- Self-play win rate swung back and forth the whole run (cycling: hiders pick a side, seekers learn it,
  hiders switch). This is why the opponent pool was added.

**Bot findings (scripted chaser seekers vs scripted fleeing hiders, tiny map, seekers' win rate):**
- 1v1: 0%. With equal speed, a hider that runs correctly laps the centre walls forever. 1v1 on the tiny map
  is a guaranteed hider win with good play.
- 2v1: about 40%, 3v1: 65%, 2v2: 22%. 2v1 is the first balanced setup, and the bots do not coordinate,
  so trained seekers that cooperate should beat 40%.

**2v1, tiny map, grid version with memory and minimap (`runs/v1.1_tiny_2v1_*`, really v1.2 rules: the run card says v1.1 because GAME_VERSION was not bumped at the time):**
- Stopped early at about 57k steps (27 updates), so not conclusive. Seekers won almost every self-play game
  early on; hiders improved against the chaser bot from being caught 81% to 65% of the time.

**Bugs and lessons found along the way:**
- Minibatch NaN crash: a leftover minibatch of size 1 made the advantage standard deviation NaN.
  Fixed by normalising advantages once per batch and splitting into equal minibatches.
- Observation blind spot: vision range 5 but the observation window reached only 4 cells, so a hider could
  be "seen" (and penalised) by a seeker it could not observe. Fixed (window 11, env asserts it).
- **Reward farming exploit (still present, fix planned for v2):** seekers get +1 every step a hider is visible,
  but catching ends the round. Following a hider at a distance for 20+ steps without catching can pay more
  than catching. Didn't show in 1v1 because seekers couldn't catch anyway.
- Grid line of sight can leak through diagonal gaps where two walls touch at a corner.

## Why v2 (continuous space with cone vision)

The grid version is not FPS-like: vision is 360 degrees with an artificial range limit, there is no facing
direction (so no flanking), sight can leak through corners, and movement is blocky. Continuous space also
is the stepping stone to the continuous-time version of the end goal.

## v2 spec (agreed)

- **Continuous positions.** Agents have a real (x, y) position, a radius and a heading. Walls come from the
  same text map files: each `#` becomes a solid 1 x 1 square. Agents slide along walls on collision.
- **Actions, two choices per step:**
  - Move: stay, forward, back, strafe left, strafe right.
  - Turn: a fixed set, suggested -90, -30, -10, -3, 0, +3, +10, +30, +90 degrees (**to confirm with Moeez**;
    the alternative he raised is a continuous turn from -180 to +180, recommended as a later experiment
    because it is harder to learn and needs a different action head).
  - Heading itself is continuous. A turn-speed limit keeps flanking meaningful.
- **Vision cone: 90 degrees**, no range limit, blocked by walls, no seeing through corners.
  - Enemy detection is **exact**: a straight line from viewer to target, checked against wall squares,
    and the target must be inside the cone. Nobody can slip between rays.
  - **Rays describe the layout:** 31 rays, one every 3 degrees across the cone. Each ray stops at the first
    wall (so nothing behind a wall is computed). Go to 46 rays (every 2 degrees) if agents miss distant
    doorways on the big map.
- **Catch: distance and facing.** Within the catch distance and the hider within about 30 degrees either
  side of where the seeker is looking (**angle to confirm**).
- **Rewards (zero-sum, hiders get exactly the opposite), fixes the farming exploit:**
  - each step a hider is in sight: seekers + 0.05 x closeness (1 when touching, fading towards 0 far away)
  - each step no hider in sight: seekers - 0.05
  - catch: + 10
  - win (all caught): + 10, plus a bonus of up to + 10 for time left (earlier catch = bigger bonus)
  - hiders survive to the end: seekers - 10
  - Distance only matters when a hider is seen. Do not reward hiders for distance while unseen (they cannot
    observe it, and it would teach running to corners instead of hiding).
- **Observations:** rays (distance and what was hit: wall, teammate, enemy), visible enemies as direction
  and distance, own heading and position, time and hiding phase, plus the **minimap** (team-shared
  "searched recently" memory now marks cells covered by the team's cones, fading over 10 steps, 1 step is
  thought of as about 1 second).
- **Keep:** 2 seekers v 1 hider, hiding phase, opponent pool, bots and benchmarks (bots need rewriting to
  steer in continuous space; they can still plan routes on the grid underneath).
- **Performance:** raycasting must be vectorised in numpy (all rays at once), numba if still slow.
  JAX later could run many games on a GPU.
- **Rendering:** draw the cone as a polygon from the ray endpoints.
- **Git:** before starting, commit the current grid code and tag it `v1-grid`, then build v2 on a branch.
  Update `GAME_VERSION`, `formulation.md` (new v2 section and changelog), README and tests together.

## v2 decisions (made 2026-10-20)

1. Turning: **fixed set of 9** (-90, -30, -10, -3, 0, +3, +10, +30, +90), as a second action head. Continuous turning
   is a later experiment.
2. Catch: within **0.8** units centre to centre, hider within **30 degrees** of where the seeker looks, **and in sight**
   (clear straight line).
3. Agent radius **0.3**, speed **1 unit per step** for forward, back and strafing alike. Moves are done in 4 pieces.
4. Smaller choices made while building (Moeez said "rest what you think is best"):
   - agents do not block each other (they can overlap), as in the grid version
   - turn first, then move; seekers frozen in the hiding phase means no moving and no turning
   - closeness = 1 - distance / 10 (clipped at 0), and the best seeing pair counts
   - the 11 x 11 close-up is dropped (rays replace it); the personal "last seen enemy" memory stays on the minimap
   - enemy slots padded to `MAX_TEAM_SIZE` (5) so bigger teams work later
   - rays stop at bodies as well as walls; "searched" marks the map cells the rays pass over (up to the wall)
   - sight is one-way now (a seeker can see a hider's back), accepted as FPS-like, documented in formulation.md

## v2 as built (branch `v2-continuous`)

- `test_env.py` has 40 checks: exact wall contact and sliding, no wall overlap in random play, rays exact,
  no seeing through corner gaps, one-way sight, cone edges, catch rules, every reward number, the anti-farming check
  (watching a hider for a whole round is worth about -7 to seekers, catching it about +30), observations, team memory.
- Speed: vision 0.5 ms per step for 3 agents; random play about 2,000 game steps/s; training about the same as the grid
  2v1 (smoke run reached 717 steps/s including the one-off bot measurements, still climbing).
- Bots rewritten: they plan on the grid of cells underneath, then turn and walk. The chaser wanders to random spots when
  it has no target; the fleeing hider waits still while looking around, then backs away from a seeker while facing it.
- v2 reference numbers (2v1 tiny, 300 games, seekers' win rate): random v random 20%, chaser v random hider 89%,
  chaser v fleeing bot 74%, random seekers v fleeing bot 4%. Not comparable with the grid bot numbers.
- Bug found while building: pushing a body out of wall squares one by one made the seam between two squares of a flat
  wall act like a corner, nudging agents sideways. Fixed by pushing out of the closest square first.
