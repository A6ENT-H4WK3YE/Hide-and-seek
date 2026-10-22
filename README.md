# Hide and Seek MARL (now: attack and defend)

An FPS-style team game (Valorant / CS style, without guns: you take an enemy out by keeping them in your sights), trained with multi-agent reinforcement learning.

Two teams of two on a map with two bomb sites, in continuous space. Attackers try to plant the spike on a site and defend it until it explodes; defenders hold the sites, retake and defuse. Every player sees only a 90 degree vision cone, hears enemies who run, plant or defuse nearby (even through walls), and shares what it knows with its teammate. A player is out once an enemy keeps them in its sights for 10 seconds in a row, so whoever spots the other first and holds the angle wins the fight.

The project started as hide and seek (v1 to v3.5); that game is kept in git under the tag `v3.5-hide-and-seek`, and what it taught us is summarised below.

## The aim

Recreate as much of an FPS game as possible, train agents on it with multi-agent RL, and study what they learn: tactics, teamwork and emergent behaviour. The model moves step by step towards being fully continuous (continuous space now, continuous time later, possibly learned dynamics such as neural ODEs), and the end point is a 3D render of the agents playing.

## The game (v4.1)

- **Map:** loaded from a text file; every `#` is a solid 1 x 1 square. Players are circles (radius 0.3) with a position and a heading, and slide along walls.
- **Actions:** every step, a move (stay, forward, back, strafe left, strafe right), a turn (-90 to +90 degrees) and whether to switch between walking and running. Running is full speed but makes noise; walking is half speed and silent. Every round starts walking.
- **Vision:** a 90 degree cone, no range limit, blocked by walls. You can see someone's back without being seen.
- **Sound:** running, planting and defusing are heard by enemies within a quarter of the map's size (about 11 cells), through walls.
- **Taking someone out:** keep an enemy in sight, within 30 degrees of where you look, for 10 steps in a row (any break restarts the count). Whoever started watching first finishes first; same-step finishes, like two players watching each other from the same moment, are a coin flip.
- **Round:** 250 steps; attackers are frozen for the first 30 (setup). An attacker standing still on a site for 5 steps plants the spike; it explodes 50 steps later unless a defender defuses it (standing still next to it for 7 steps).
- **Winning:** attackers win if the spike explodes or every defender is out; defenders win by defusing, by taking every attacker out before the plant, or when time runs out before a plant.
- **Rewards (zero-sum):** +10 per takedown, +5 for the plant, +20 / -20 for the round, plus a small potential-based shaping term that guides attackers towards the sites without changing which strategies are best.

Full rules, and the reasoning behind every choice, are in `formulation.md`.

## What each player knows

- **Rays:** 31 rays across its cone, each saying how far it reaches and whether it hit a wall, a teammate or an enemy.
- **Enemies:** every enemy its team sees or hears right now (team callouts): direction, distance, whether I see it, whether a teammate does, whether it was only heard, how far my team is towards taking it out, whether it is planting.
- **Teammates:** direction, distance, how far it is towards taking an enemy out, planting, sees an enemy.
- **Numbers:** position, heading, time, setup phase, running, my own progress towards a takedown, plant progress, the spike (down, time left, direction, distance, defuse progress), players left on each team.
- **Minimap:** the whole map shrunk to 12 x 12: walls, sites, itself, teammates, where its team searched recently, where its team last saw or heard enemies.
- **Memory:** both teams have a learned memory (a GRU) carried through each round.

## Training

- **PPO**, one network per team shared by its players, both teams learning at once (self-play), discount 0.997.
- **Head start by imitation:** before self-play, each team's network copies the scripted bots (`pretrain.py`), so attackers already know how to walk to a site and plant. Trained from scratch, they never found the plant on their own.
- **Opponents:** in 20% of games one team is played by the scripted bots; in half of the rest, one team is a past version of itself from an opponent pool that spans the whole run. Only current selves learn. This stops either side winning by exploiting one particular opponent.
- **Parallel games:** the games run in 16 worker processes, ray casting is compiled with numba, and the network stays on the CPU (faster than a GPU for a network this small). About 1,100 steps per second on a 15-core Mac.

```
pip install -r requirements.txt               # plus torch (see setup notes)
python test_env.py                            # check the rules (56 checks, should say: all checks passed)
python random_demo.py                         # bots v bots: one GIF plus win rates
caffeinate -i python train.py smoke 20000     # quick check that training runs
caffeinate -i python train.py                 # full run (TOTAL_STEPS in config.py)
python evaluate.py runs/<run name>/checkpoints/latest.pt
```

Each run gets its own folder in `runs/` with a `run_card.md` (every setting, game version, git commit), `metrics.csv`, `curves.png`, `bots.csv`, recorded GIFs and checkpoints. Ctrl+C stops training early and still saves.

**Reading the GIFs:** attackers are red, defenders blue. Shaded wedges are vision cones; dashed lines show who sees whom. A dashed ring means the player made noise; the faint dotted circle shows how far it carries, and a yellow dotted line joins every player to the enemy noise it hears. An orange ring means an enemy has that player in its sights (the thicker, the closer to being taken out), a grey X means out, and the yellow star is the spike. GIFs play at 3 frames per second.

## Progress

| Version | What it added | Status |
|---|---|---|
| v1 | grid world, 360 degree vision, hide and seek | done (tag `v1-grid`) |
| v1.1 / v1.2 | bigger observation window; team-shared "searched recently" memory and a minimap | done |
| v2 / v2.1 | continuous space, 90 degree vision cone, rays, exact line of sight, catch by distance and facing | done |
| big map | `three_sites.txt` (3 sites, mid, lanes), 250-step rounds | done, the default |
| v3 to v3.2 | walk or run, footsteps heard through walls, walk / run as a mode you switch | done |
| v3.3 to v3.5 | hiders retrained, team search memory of a minute, learned memory (GRU), league training against a cheating escape bot | done (tag `v3.5-hide-and-seek`) |
| v4 | attack and defend: two-tag takedowns, the spike (plant, explode, defuse), team callouts, memory for both teams, bots as training opponents | replaced by v4.1 |
| **v4.1** | takedowns by keeping an enemy in your sights for 10 steps; balanced, mirror-symmetric two-site map with no spawn-to-spawn sightlines | **current**, training |

Next: watch what v4 learns (rotations, trades, retakes), then bigger teams towards 5v5, continuous turning, communication between teammates, continuous time, and a 3D render.

## What hide and seek taught us (v1 to v3.5)

- **Whichever team can never win stops learning.** It happened to seekers in 1v1 and to hiders again and again. Balance matters as much as the algorithm.
- **Agents do exactly what the reward pays for.** With a small reward for keeping a hider in view, seekers spun like lighthouses instead of chasing; anywhere in the cone paid the same, so they never centred the target.
- **A predictable player loses, however good.** Hiders learned to be completely silent, then all hid at site A, and the seekers went straight there.
- **Beating your training partner is not the same as being good.** Seekers that won 84 to 99% against their own hiders caught a scripted escape bot only 27% of the time: they had learned their opponents' habits. Fixed outside opponents are the honest test, which is why v4 trains against bots too.
- **The discount has to fit the round.** With 250-step rounds and 0.99, a win at the end was worth a tenth of its points at the start.
- **Teamwork needs information.** Shared rewards alone gave almost no cooperation; a teammate cannot help with a chase it does not know about. v4 shares what each team sees and hears, and makes takedowns need two players.
- **For a small network, the game is the bottleneck.** The Mac GPU was slower than the CPU; parallel game workers and compiled ray casting made training 4 to 5 times faster.

Hide-and-seek results (seekers' win rate, 300 games per matchup; "chaser" and "fleeing bot" are scripted benchmarks):

| Version, map | Trained seekers v random hider | Trained seekers v fleeing bot | Chaser bot v trained hiders (lower = better hiders) | Self-play |
|---|---|---|---|---|
| v2, tiny | 65% | 21% (chaser 79%) | 91% (random hider 88%) | 41% |
| v2.1, big map | 77% | 37% (chaser 51%) | **54%** (random hider 70%) | 37% |
| v3, big map | 71% | **74%** (chaser 80%) | 94% (random hider 96%) | 85% |
| v3.4, big map | 84% | 53% (chaser 91%) | **67%** (random hider 100%) | 97% |
| v3.5, big map | 96% | **89%** (chaser 91%) | 94% (random hider 100%) | 84% |

## Maps

Maps are plain text files in `maps/`: `#` wall, `.` floor, `h` defender spawn, `s` attacker spawn, `A` `B` `C` site cells. `python preview_map.py maps/three_sites.txt` draws a map and checks every walkable cell is reachable.

- **`two_sites.txt`** (45 x 31): the default. Two sites, mirror symmetric, a mid room linking both sites, side doors into mid from each spawn (the spawns cannot see each other), outer lanes and a lower-mid route. Defenders spawn at the top (12 steps from a site), attackers at the bottom (22 steps).
- **`three_sites.txt`** (45 x 31): the older three-site layout used for hide and seek and the first v4 run.
- **`tiny.txt`** (11 x 11): small debugging map from the hide-and-seek days (it has no sites).

## Repo layout

```
config.py      all settings in one place
env.py         the game: movement, vision cones, hearing, tags, the spike, rewards, observations
geometry.py    moving round players among square walls, and casting rays (compiled with numba)
bots.py        scripted attackers and defenders: fixed benchmarks and training opponents
model.py       TeamNet: one network per team, with a learned memory (GRU)
train.py       PPO self-play training loop, opponent pool, bots, recordings
workers.py     run the training games in parallel processes, so every CPU core is used
matches.py     play any mix of trained / bot / random teams
evaluate.py    play a checkpoint against random players and the bots
render.py      draw frames, save GIFs
test_env.py    checks that the game rules work
random_demo.py bots or random players: one GIF plus win rates and speed
pretrain.py    imitation pretraining: both teams first copy the scripted bots, then self-play takes over
preview_map.py draw a map to PNG and check it is fully connected
formulation.md the exact game definition for each version, changelog and experiment notes
runs/          training outputs (ignored by git)
```

## Setup notes

PyTorch is installed separately because the right build depends on your machine:

- **Mac:** `pip install torch`. Training runs on the CPU (`DEVICE = "cpu"` in config): for this small network it is faster than the Mac GPU.
- **NVIDIA:** the CUDA build from pytorch.org. **AMD on Windows:** AMD's PyTorch on Windows build (ROCm).
- **No GPU / Codespaces:** `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- `pip install numba` makes the game about 3 times faster (it runs without it, just slower).

## Licence

MIT
