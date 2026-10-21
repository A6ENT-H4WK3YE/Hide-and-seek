# CLAUDE.md: Hide and Seek MARL

Read this first. It is the project memory: what we are building, what has happened so far, what we learned,
what comes next, and how Moeez wants to work. Keep it up to date when decisions are made or results come in.
The exact rules of every game version are in `formulation.md`; this file is the story and the state of play.

## Where we left off (2026-10-21, late night)

- Branch `main`, game version **v3.5**, map `three_sites.txt`, 2 seekers v 1 hider, 16 game workers, numba ray casting.
- **v3.5 run done and analysed** (results below), both teams learned: seekers from the v3.4 seekers, hiders from the v3.4 hiders plus a new
  learned memory (GRU, `MEMORY_TEAMS = ["hiders"]`). Also in this run: no hider noise cost (zero-sum again), and the
  win / loss reward doubled to 20 (catch about +40 v watching a whole round about +20). Run name `v3.5_both_*`.
- What to check after it: do hiders vary their spot now (heatmap, share of rounds at A / B / C), do they run away when
  they hear a seeker (run share and distance change when hearing), seekers' coverage and dead zones, who wins,
  evaluate.py. Analysis scripts in the scratchpad need updating: networks now come from `model.load_net`, and `act`
  takes and returns the memory (`a, logp, value, h = net.act(obs, h)`).
- Not taken: a random "plan" number for hiders each round (Moeez: let them learn it through memory); a flee reward
  (would be reward tweaking; removing the noise cost instead).
- Still queued: potential-based exploration shaping for seekers if dead zones remain; team spotting; aim reward;
  a sharper minimap (12 x 12 blurs the 1-wide corridors: hiders bump into walls on 10% of moves, seekers 4%).
- **Always start runs with `caffeinate -i`, smoke runs and evaluations too.**

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
  If several changes must go together (or he asks for them together), say so and note it in `formulation.md`.
- **Be direct and concise.** He dislikes long question lists; when there are choices, give a recommendation.
- **He learns by understanding, not just receiving code.** Explain what each new piece does and why.
- **Check the actual files before answering** about the code or results. Build on the existing code, do not
  rewrite from scratch.
- **Test before handing over:** run `python test_env.py` and a smoke run (`python train.py smoke 20000`)
  after any change to the game or training code. Then the full run, `evaluate.py`, and a look at behaviour.
- **He watches the GIFs** and spots behaviour problems there (spinning seekers, no teamwork, hiders not looking
  around). Back his observations with measurements before proposing a fix.
- He likes being told what is slow and why; commit before long runs so the run card records a clean commit.

## Machine setup

- MacBook (university-managed), **Apple M5 Pro, 15 cores** (5 performance, 10 efficiency). Python 3.9.6 in `.venv`
  (`source .venv/bin/activate`). **Keep code Python 3.9 compatible:** no `X | Y` type hints, no `match` statements.
- Training on **CPU** (`DEVICE = "cpu"`). Measured on v3.1: the Mac GPU (MPS) is 16x slower at choosing actions
  (tiny network, only 32 agents at a time) and no faster at learning.
- **Games run in parallel worker processes** (`workers.py`, `N_WORKERS = 16`, one per game). Game stepping is 6x faster
  than one process, with identical results. Training on the big map: about 2,300 steps/s, a 5M run about 45 minutes.
  The bot benchmarks at each recording still run in one process.
- Long runs: `caffeinate -i python train.py` so the Mac does not sleep. The screen may turn off; do not close the lid,
  quit VS Code or let the battery run out. To stop a run and keep a checkpoint, send Ctrl+C (SIGINT), not a kill.
- GitHub repo: `A6ENT-H4WK3YE/Hide-and-seek`. There is also a GitHub Codespace copy (2 cores, slow);
  the Mac is the main machine. Commit and push before switching machines.
- **Ray casting is compiled with numba** (`geometry.py`; falls back to the numpy version if numba is missing).
  It was 88% of each game step: about 25 cell steps per call, each ~20 tiny numpy calls on only ~100 rays, so numpy's
  per-call overhead dominated. Compiled: 46x faster rays, game step 2.07 -> 0.51 ms (measured on the desktop).
- He also has a **Windows desktop** (AMD Ryzen 5 7500X3D, 6 cores, RX 9070 GRE), set up 2026-10-21: Python 3.13 in
  `.venv`, CPU torch, activate with `.\.venv\Scripts\Activate.ps1`. Smart App Control is on and checks each new DLL
  the first time it loads (the first numpy / torch import can fail or take minutes; later ones are fine).
  Measured there: training about 1,650 steps/s with numba (860 without), bot games 0.12 to 0.36 s each.
  GPU (ROCm) and overclocking were considered and rejected: the game, not the network, is the bottleneck.
  Not yet tried: `N_WORKERS` = core count (16 workers share 6 cores there).

## The end goal

Recreate as much of an FPS game as possible (CS / Valorant style, no shooting yet), train agents on it
with multi-agent reinforcement learning, and study what they learn to do: tactics, teamwork, emergent
behaviour. Move towards a fully continuous model (continuous space now, continuous time later, possibly
learned dynamics such as neural ODEs, which ties into his PhD). Finish with a **3D render** showing the
agents playing in a 3D world. The goal can grow later.

Roadmap (status):
1. v1: grid world, 360 degree vision. **Done** (tag `v1-grid`).
2. v2: continuous space, 90 degree vision cone, catch by distance and facing. **Done.**
3. v3: walk vs run with sound. **Built; v3.2 (sound through walls) is current; hiders still need to learn to stay quiet.**
4. Bigger map and bigger teams. **Big map done** (`three_sites.txt` is the default since v2.1); bigger teams not yet.
5. Team spotting and aim (next), then continuous turning, learned memory (GRU), communication between teammates.
6. Continuous time (positions follow differential equations).
7. 3D render (the vision rays can drive a Wolfenstein-style 3D view).

## Repo layout and git

```
config.py       all settings in one place (GAME_VERSION, map, bodies, vision, sound, rewards, training, pool, workers)
geometry.py     round agents sliding along square walls, ray casting (all rays at once, cell by cell, exact)
env.py          the game: movement, vision cones, hearing, catches, rewards, observations, memory, minimap
render.py       draws frames and GIFs at 3 fps (cones, heading ticks, sight lines; sound: dashed ring = running,
                dotted circle = how far the footsteps carry, yellow dotted line = who hears whom)
model.py        TeamNet: one network per team, shared by teammates; three action heads (move, turn, walk/run switch);
                optional learned memory (GRU) for teams in MEMORY_TEAMS
train.py        PPO self-play, opponent pool, run card, metrics, GIFs, checkpoints, bot benchmarks
workers.py      runs the training games in parallel processes (one per CPU core)
evaluate.py     plays a checkpoint against random players and the bots
bots.py         scripted chaser seeker and fleeing hider (fixed benchmarks; plan on the grid, steer continuously)
matches.py      plays any mix of trained / bot / random players
test_env.py     56 rule checks (must say "all checks passed")
random_demo.py  random agents: one GIF plus baseline win rates and speed
preview_map.py  draws a map file to PNG and checks it is fully connected (note: it overwrites the tracked preview PNG)
formulation.md  the exact game definition per version, changelog and experiment notes
maps/           tiny.txt (11 x 11, debugging), three_sites.txt (45 x 31, FPS-style, 3 sites)
runs/           training outputs, ignored by git (run_card.md, metrics.csv, curves.png, bots.csv, gifs/, checkpoints/,
                and for analysed runs behaviour_heatmaps.png)
```

Git: `main` = the latest work (v2 to v3.1 and on). The grid version is kept as tag `v1-grid`; tag `v3.1` marks the end of
the 2026-10-21 session. Branch `v2-continuous` was the working branch while v2 was built (same history, no longer needed).

Commands:
```
python test_env.py
python train.py smoke 20000          # quick check
caffeinate -i python train.py        # full run (TOTAL_STEPS in config)
python evaluate.py runs/<run>/checkpoints/latest.pt 300
```

## The game now (v3.5, details in formulation.md)

- Continuous 2D space. Every `#` of the map is a solid 1 x 1 square. Agents are circles (radius 0.3) with a position
  and a heading; they slide along walls and can overlap each other. 1 unit = 1 cell, 1 step = about 1 second.
- Actions each step: move (stay, forward, back, strafe left, strafe right), turn (-90, -30, -10, -3, 0, +3, +10, +30,
  +90 degrees), and keep / switch the walk-run mode. Run = 1 unit per step and noisy; walk = 0.5 and silent.
  Every round starts walking.
- Vision: 90 degree cone, no range limit, walls block it; enemies checked exactly (straight line between centres);
  31 rays describe the layout. Sight is one-way (you can see someone's back).
- Hearing: an enemy that runs is heard within 0.25 x the map's longer side (11.25 cells on three_sites), straight line,
  through walls (v3.2; v3 and v3.1 used 6 cells of walking distance, which walls blocked).
- Catch: in sight, centres within 0.8, hider within 30 degrees of where the seeker looks.
- Rewards (seekers; hiders get exactly minus): +0.2 x closeness per step a hider is in sight, -0.05 per step none is,
  +10 per catch, +20 plus up to +10 for time left when all are caught, -20 if hiders survive (win / loss doubled in
  v3.5). v3.3 and v3.4 also charged hiders 0.02 per noisy step; removed in v3.5, so it is zero-sum again.
- Round: 250 steps with a 30-step hiding phase on the big map (100 and 15 on the tiny map). 2 seekers v 1 hider.
- Each agent observes: rays, enemies in sight, enemies heard, 7 numbers (position, heading, time, hiding phase,
  running) and a 12 x 12 minimap (walls, me, teammates, team "searched recently" fading over 60 steps since v3.4, own
  "last seen enemy" memory).
- Training: PPO, one network per team. Normally both teams learn (self-play), with half the games against frozen past
  versions from a pool that spans the whole run. `TRAIN_TEAMS` can freeze a team: it then plays with an earlier run's
  networks (`FROZEN_RUN`), its final network plus its checkpoints as the pool. The walk / run switch starts at 95% "keep".

## How we got here: versions, ideas and fixes

| Version | What changed | Why |
|---|---|---|
| v1 | grid world, 360 degree vision range 5, adjacent catch, +-1 per step seen/hidden | first working game |
| v1.1 | observation window 9 x 9 -> 11 x 11 | a hider could be "seen" by a seeker it could not observe |
| v1.2 | team-shared "searched recently" memory and a 12 x 12 minimap (first labelled "v2", renamed) | seekers could not tell where they had already searched |
| v2 | continuous space, 90 degree cone, rays, exact sight, catch by distance + facing, new rewards | grid was not FPS-like; old rewards could be farmed |
| v2.1 | reward for a hider in sight 0.05 -> 0.2 per step | seekers learned to spin instead of locking on |
| (big map) | `three_sites.txt`, 250-step rounds, 30-step hiding phase, 5M steps | Moeez wanted to see emergent behaviour on a real map |
| v3 | walk (half speed, silent) or run (full speed, heard within 6 cells of walking distance) | Moeez: hiders could not tell a seeker was behind them |
| v3.1 | walk / run becomes a mode you switch, starting in walk; opponent pool spans the whole run (together, at Moeez's request) | hiders never learned to stay quiet |
| (speed) | games in 16 parallel worker processes | training was slow; the game was the bottleneck, not the network |
| (speed) | ray casting compiled with numba, identical results | profiling showed rays were 88% of each game step |
| v3.2 | sound goes through walls; range = 0.25 x the map's longer side (11.25 on three_sites) | Moeez: with 6 cells of walking distance sound almost never reached anyone, so hiders had no idea where seekers were |
| v3.3 | hiders pay 0.02 per noisy step; only hiders learn, against frozen v3.2 seekers (final + 17 checkpoints as pool); switch starts at 95% "keep" (all together, agreed) | v3.2 seekers won 98%; hiders flipped walk / run constantly and were heard through walls. Moeez: fix the hiders first |
| v3.4 | team "searched recently" memory fades over 60 steps instead of 10 (no reward change); both teams learn from v3.2 seekers / v3.3 hiders | Moeez: seekers leave dead zones and forget where they searched; give them the information rather than an exploration reward |
| v3.5 | hiders get a learned memory (GRU, starts switched off); noise cost removed; win / loss reward 10 -> 20 (together, at Moeez's request) | v3.4 hiders were silent but always at site A, and never ran away when they heard a seeker |

Decisions behind v2 (agreed 2026-10-20): fixed set of 9 turns (continuous turning is a later experiment); catch 0.8 and
30 degrees and in sight; radius 0.3, speed 1 for every direction; agents pass through each other; turn then move;
closeness = 1 - distance / 10; the 11 x 11 close-up replaced by rays; enemy slots padded to 5 for bigger teams later;
one-way sight accepted as FPS-like.
Decisions behind v3 (2026-10-21): no reward for "seeing a seeker and running away" (could be exploited); walk at 0.5
(like CS; 0.75 would make walking nearly free); hearing range 6 cells of walking distance (3 x 3 or 4 x 4 would only
let you hear a seeker already in catch range); only enemies' footsteps are reported; heard enemies go into memory;
bots hear too. Revised in v3.2 (2026-10-21, Moeez): sound must go through walls, and the range must
reach from the middle of the map to the hider spawn, scaling with map size; chose a quarter of the longer side.

## Results so far (seekers' win rate, 300 games per matchup unless noted)

| Run | Map, steps | Trained seekers v random hider | Trained seekers v fleeing bot | Chaser bot v trained hiders (lower = better hiders) | Self-play |
|---|---|---|---|---|---|
| v2 | tiny, 2M | 65% (random seekers 22%) | 21% (chaser 79%) | 91% (random hider 88%) | 41% |
| v2.1 | tiny, 2M | 96% | 67% (chaser 79%) | 91% (random hider 88%) | 86% |
| v2.1 | big, 5M | 77% (random seekers 1%, chaser 70%) | 37% (chaser 51%) | **54%** (random hider 70%, fleeing bot 51%) | 37% |
| v3 | big, 5M | 71% (chaser 96%) | **74%** (chaser 80%) | 94% (random hider 96%, fleeing bot 80%) | 85% |
| v3.1 | big, 3.4M (stopped) | 84% (chaser 96%) | 41% (chaser 80%) | **87%** (random hider 96%, fleeing bot 80%) | 31% |
| v3.2 | big, 5M | 99% (chaser 100%) | 76% (chaser 91%) | 99% (random hider 100%, fleeing bot 88%) | 98% |
| v3.3 | big, hiders only v frozen v3.2 seekers, stopped at 1.85M | (seekers frozen) | (seekers frozen) | not evaluated | 1.5% v the final v3.2 seekers |
| v3.3 both | big, both learning, seekers from v3.2, hiders from v3.3, stopped at 213k | not evaluated | | | 2% -> 10% |
| v3.4 both | big, 5M, seekers from v3.2, hiders from v3.3, team memory 60 steps | 84% (chaser 100%) | 53% (chaser 91%) | **67%** (random hider 100%, fleeing bot 91%) | 97% |
| v3.5 both | big, 5M, from v3.4, hider memory, no noise cost, win / loss 20 | 96% (chaser 100%) | **89%** (chaser 91%) | 94% (random hider 100%, fleeing bot 91%) | 84% |

Bot numbers change with each version (in v3 the bots also hear), so compare trained teams with the bot line of the same row.

Earlier, on the grid: in 1v1 (v1, 2M steps) hiders learned to run (random seekers won 1.2% against them) and seekers
reached 50% against random hiders, but trained v trained ended 0% and seekers gave up. Scripted bots showed 1v1 on the
tiny map is always a hider win; 2v1 (about 40%) was the first balanced setup.

What the agents did (measured, not just watched):
- **v2 tiny:** seekers strafed right while spinning (86% of moves strafe right, most common turn +90), sweeping the room
  like a lighthouse; with a hider in sight they never walked forward. The two seekers did split up to search.
- **v2.1 tiny:** spinning gone; with a hider in sight they walk forward 74% of the time. New quirk: while searching
  they walked backwards (75% of moves), since backpedalling cost nothing.
- **v2.1 big map:** both teams learned. Hiders stay near spawn in the hiding phase, then rotate **left to site A**
  (31% of hunting time, plus 40% in nearby corridors; site B in the middle only 3%). Seekers push **straight up the
  middle**, then spread to the sites, a median 10 units apart. Most catches around A.
- **v3 big map:** seekers run 92% of the time, walk forward 76% of moves (strafing and backpedalling gone), lock on
  99% of the time when they see the hider, and heard the hider without seeing it in 2,262 steps of 150 games.
  Hiders run 79% of the time, keep moving 91% even with nobody around, and when they hear a seeker they always run,
  but it still closes in. Hiders switched to **site C** and the right-hand corridors. **Forced-walk test:** the same
  trained hiders forced to walk cut the seekers' self-play win rate from 77% to 48%.
- **v3.1 (partial, 3.4M steps, evaluated under its own hearing rule of 6 cells walking):** the most balanced run so far.
  Self-play 31%, the chaser bot catches trained hiders 87% (random hider 96%); trained seekers beat the fleeing bot only
  41%. Both teams were in run mode about half the time (hiders press "switch" on 37% of steps). Forcing hiders to walk
  no longer helped (43% -> 52% seeker wins), so their running was not the main weakness any more.
- **v3.2 big map (sound through walls):** seekers took off after about 1M steps and won 98% of self-play. They learned to
  hold their walk / run mode (switch on 13% of steps, in run mode 54%), keep a seen hider in view 89% of the next steps
  (v3: 74%) and travel as a pair (median 5.7 units apart; v3: 10.8). Hiders flip their mode constantly (switch on 61% of
  steps, making noise 46% of hunting steps) and are caught by the chaser bot 99% (random hider 100%). Seekers heard the
  hider without seeing it in 1,747 steps of 150 games. When a hider hears a seeker it always moves, but the seeker
  closes in (-0.3 units per step). Forced-walk test: 99% -> 89%. Hiders spend the hunt in corridors (57%), site C (27%),
  never site A. Still no teamwork: while one seeker sees the hider, the other walks +0.06 cells closer per step
  (when nobody sees it, +0.36).
- **v3.4 both teams learning (`runs/v3.4_both_three_sites_2v1_20261021_193635`):** hiders are completely silent (walk
  100% of the time, never heard) and the best hiders so far against the chaser bot (67% caught; random 100%, fleeing bot
  91%). But every hider goes to **site A** along the top corridor (38% of the hunt on A, 48% in nearby corridors, 0% on
  B and C), and the seekers learned it within about 400k steps: they run up the middle straight to A (forward 89% of
  moves, keep a seen hider in view 92%), travel as a pair (median 5 units apart) and never search C or the lanes. Self-play
  97 to 99% for seekers. Specialising cost them generality: against random hiders they win only 84% (v3.2: 99%), against
  the fleeing bot 53% (v3.2: 76%). Some teamwork appeared: while one seeker sees the hider the other walks +0.33 cells
  closer per step (v3.2: +0.06). Dead zones remain, now because the hider is predictable, not because of memory.
- **v3.5 (`runs/v3.5_both_three_sites_2v1_20261021_220610`): hiders vary their side now, but are loud again.**
  Side the hider mostly stays on per round (200 self-play rounds): v3.4 left / A 90%, middle 10%, right / C 0%;
  v3.5 left / A 12% (hiders win 88% there), middle / B 24% (win 17%), right / C 64% (win 1%). So they mix, but mostly
  pick the side the seekers now cover. Without the noise cost they are in run mode 59% of hunting steps (v3.4: 0%) and
  seekers heard them without seeing them in 3,131 steps (v3.4: 0); forced to walk, the seekers' win rate drops 90% ->
  79%. When a hider hears a seeker it now runs 62% of the time; the seeker still closes in, but slower (-0.24 per step,
  v3.4: -0.53). Seekers became much more general: 89% against the fleeing bot (v3.4: 53%, chaser bots 91%), 96%
  against random hiders; they run 93% of the time, 92% of moves forward, cover C, mid and the top corridor, and now
  leave site A thin. No teamwork yet (other seeker +0.02 cells closer per step while its mate sees the hider).
  Hider entropy rose during training (1.5 -> 2.6): less decisive, more varied.
- **Seeker problems Moeez spotted (v3, measured):** sloppy lock-on (they turn towards a seen hider 87% of the time but
  pick the centring turn only 39%; the hider drifts from 19 to 30 degrees off-centre per step and is lost 26% of the
  time) and no teamwork (while one seeker sees the hider, the other walks 0.07 cells closer per step, less than its
  0.15 when nobody sees it: it has no way of knowing).

## Insights so far

- **Learned helplessness is the recurring problem.** Whichever team can never win stops learning, because every action
  ends the same way: seekers in v1 1v1, hiders on the tiny map in v2, hiders again in v3 once seekers took off.
  Balance (team sizes, map, information) matters as much as the algorithm.
- **The map decides whether hiding is learnable.** On the tiny map a searching seeker finds you almost anywhere, so
  hiders never learned; on the big map they did (v2.1), and picked a site.
- **Agents do exactly what the reward pays for.** With 0.05 per step for keeping a hider in view, spinning was good
  enough (a sweeping seeker lands a catch when the hider falls into its 30 degree window). At 0.2 they locked on.
  Anywhere in the cone pays the same, so they do not bother to centre the target (the aim reward is for that).
- **Free movement rules create odd habits.** With equal speed in every direction, seekers strafed and backpedalled;
  once running and sound existed, they ran forward instead, without any change aimed at that.
- **A toggle only helps if the untrained network actually holds it.** A fresh network presses "switch" half the time,
  so the mode flips almost every step. Seekers (who win and get clear feedback) learned to hold their mode; hiders
  never did, so they kept making noise at random.
- **Sound through walls helps whoever is hunting.** Two seekers listening for one hider, plus a hider that cannot
  control its noise, turned a balanced game (v3.1, 31%) into a seeker walkover (v3.2, 98%).
- **A new mechanic can help one side far more.** Sound mostly helped seekers: they learned to listen, hiders never
  learned to be quiet. The forced-walk test showed the game was fine and the learning was the problem, which led to
  the walk toggle (one decision keeps you quiet, instead of a fresh coin flip every step).
- **Teams pick a side, and it moves.** Hiders favoured site A in v2.1 and site C in v3; seekers learn where hiders go.
  This is the cycling the opponent pool exists to damp, and why the pool now spans the whole run.
- **A predictable hider loses, however quiet.** Once hiders were silent they all settled on site A, the seekers read it
  and won 99%. The best hider play mixes sites at random; PPO self-play tends to collapse onto one plan.
- **Teamwork needs information.** Shared rewards alone gave no cooperation beyond loosely splitting up; the second
  seeker cannot help with a chase it does not know about (hence team spotting).
- **Measure behaviour, not just win rates.** Move and turn counts, "walks forward when it sees the hider",
  forced-action tests and heatmaps found every real problem; the curves alone did not.
- **Per-step rewards scale with round length.** The anti-farming margin that was safe in 100-step rounds almost
  vanished in 250-step rounds (+30.0 for watching v +30.1 for catching, undiscounted; with the 0.99 discount the
  agents compare about +15 v +30, so catching still wins).
- **For a small network, the game is the bottleneck, not the network.** The GPU was slower; parallel game workers
  made training 2.6x faster.

## Bugs and lessons found along the way

- Minibatch NaN crash: a leftover minibatch of size 1 made the advantage standard deviation NaN.
  Fixed by normalising advantages once per batch and splitting into equal minibatches.
- Observation blind spot (grid): vision range 5 but the window reached only 4 cells. Fixed in v1.1.
- Reward farming exploit (grid): +1 per step a hider was visible paid more than catching after about 26 steps.
  Fixed in v2 with small per-step rewards; margin needs fixing again for 250-step rounds (queued).
- Grid line of sight leaked through diagonal gaps where two walls touch at a corner. Fixed in v2: a ray exactly
  through such a corner counts as blocked.
- Wall sliding bug (v2 build): pushing a body out of wall squares one by one made the seam between two squares of a
  flat wall act like a corner, nudging agents sideways. Fixed by pushing out of the closest square first.
- Version label clash: the minimap change had been called "v2" in formulation.md while config said v1.1; renamed to
  v1.2 so v2 could be continuous space. The run `runs/v1.1_tiny_2v1_*` really used v1.2 rules.
- `docs/formulation.md` paths pointed to a folder that never existed; fixed.
- Python's multiprocessing on a Mac re-runs `train.py` inside every worker (it has no main guard), so the workers are
  plain subprocesses running `workers.py`, talking through pipes.
- `preview_map.py` rewrites the tracked preview PNG; restore it with `git restore` if nothing changed in the map.
