# Hide and Seek MARL (now: attack and defend)

An FPS-style team game (Valorant / CS style, with short-range "tags" instead of guns), trained with multi-agent reinforcement learning.

Two teams of two on a map with three bomb sites, in continuous space. Attackers try to plant the spike on a site and defend it until it explodes; defenders hold the sites, retake and defuse. Every player sees only a 90 degree vision cone, hears enemies who run, plant or defuse nearby (even through walls), and shares what it knows with its teammate. Taking someone out needs two teammates: a player is out once tagged by two different enemies within a few seconds.

The project started as hide and seek (v1 to v3.5); that game is kept in git under the tag `v3.5-hide-and-seek`, and what it taught us is summarised below.

## The aim

Recreate as much of an FPS game as possible, train agents on it with multi-agent RL, and study what they learn: tactics, teamwork and emergent behaviour. The model moves step by step towards being fully continuous (continuous space now, continuous time later, possibly learned dynamics such as neural ODEs), and the end point is a 3D render of the agents playing.

## The game (v4)

- **Map:** loaded from a text file; every `#` is a solid 1 x 1 square. Players are circles (radius 0.3) with a position and a heading, and slide along walls.
- **Actions:** every step, a move (stay, forward, back, strafe left, strafe right), a turn (-90 to +90 degrees) and whether to switch between walking and running. Running is full speed but makes noise; walking is half speed and silent. Every round starts walking.
- **Vision:** a 90 degree cone, no range limit, blocked by walls. You can see someone's back without being seen.
- **Sound:** running, planting and defusing are heard by enemies within a quarter of the map's size (about 11 cells), through walls.
- **Tags:** an enemy in sight, within 3 units and within 20 degrees of where you look gets tagged. One tag only marks them; tags from two different enemies within 8 steps take them out. A lone survivor cannot take anyone out alone.
- **Round:** 250 steps; attackers are frozen for the first 30 (setup). An attacker standing still on a site for 5 steps plants the spike; it explodes 40 steps later unless a defender defuses it (standing still next to it for 7 steps).
- **Winning:** attackers win if the spike explodes or every defender is out; defenders win by defusing, by taking every attacker out before the plant, or when time runs out before a plant.
- **Rewards (zero-sum):** +2 per tag, +10 per takedown, +5 for the plant, +20 / -20 for the round, plus a small potential-based shaping term that guides attackers towards the sites without changing which strategies are best.

Full rules, and the reasoning behind every choice, are in `formulation.md`.

## What each player knows

- **Rays:** 31 rays across its cone, each saying how far it reaches and whether it hit a wall, a teammate or an enemy.
- **Enemies:** every enemy its team sees or hears right now (team callouts): direction, distance, whether I see it, whether a teammate does, whether it was only heard, whether my team tagged it recently, whether it is planting.
- **Teammates:** direction, distance, tagged recently, planting, sees an enemy.
- **Numbers:** position, heading, time, setup phase, running, tagged recently, plant progress, the spike (down, time left, direction, distance, defuse progress), players left on each team.
- **Minimap:** the whole map shrunk to 12 x 12: walls, sites, itself, teammates, where its team searched recently, where its team last saw or heard enemies.
- **Memory:** both teams have a learned memory (a GRU) carried through each round.

## Training

- **PPO**, one network per team shared by its players, both teams learning at once (self-play), discount 0.997.
- **Opponents:** in 20% of games one team is played by the scripted bots; in half of the rest, one team is a past version of itself from an opponent pool that spans the whole run. Only current selves learn. This stops either side winning by exploiting one particular opponent.
- **Parallel games:** the games run in 16 worker processes, ray casting is compiled with numba, and the network stays on the CPU (faster than a GPU for a network this small). About 1,100 steps per second on a 15-core Mac.

```
pip install -r requirements.txt               # plus torch (see setup notes)
python test_env.py                            # check the rules (59 checks, should say: all checks passed)
python random_demo.py                         # bots v bots: one GIF plus win rates
caffeinate -i python train.py smoke 20000     # quick check that training runs
caffeinate -i python train.py                 # full run (TOTAL_STEPS in config.py)
python evaluate.py runs/<run name>/checkpoints/latest.pt
```

Each run gets its own folder in `runs/` with a `run_card.md` (every setting, game version, git commit), `metrics.csv`, `curves.png`, `bots.csv`, recorded GIFs and checkpoints. Ctrl+C stops training early and still saves.

**Reading the GIFs:** attackers are red, defenders blue. Shaded wedges are vision cones; dashed lines show who sees whom. A dashed ring means the player made noise; the faint dotted circle shows how far it carries, and a yellow dotted line joins every player to the enemy noise it hears. An orange ring means tagged recently, a grey X means out, and the yellow star is the spike. GIFs play at 3 frames per second.

## Progress

| Version | What it added | Status |
|---|---|---|
| v1 | grid world, 360 degree vision, hide and seek | done (tag `v1-grid`) |
| v1.1 / v1.2 | bigger observation window; team-shared "searched recently" memory and a minimap | done |
| v2 / v2.1 | continuous space, 90 degree vision cone, rays, exact line of sight, catch by distance and facing | done |
| big map | `three_sites.txt` (3 sites, mid, lanes), 250-step rounds | done, the default |
| v3 to v3.2 | walk or run, footsteps heard through walls, walk / run as a mode you switch | done |
| v3.3 to v3.5 | hiders retrained, team search memory of a minute, learned memory (GRU), league training against a cheating escape bot | done (tag `v3.5-hide-and-seek`) |
| **v4** | attack and defend: two-tag takedowns, the spike (plant, explode, defuse), team callouts, memory for both teams, bots as training opponents | **current**, training |

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

- **`three_sites.txt`** (45 x 31): three sites, a mid room, two long lanes, rotation links and cover boxes. Defenders spawn at the top, attackers at the bottom. The default map.
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
