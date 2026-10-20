# Game formulation

The game is a **partially observable stochastic game (POSG)**: several agents act at the same
time, the game moves on according to their joint action, and each agent only sees part of what
is going on. This file says exactly which game each version is. `GAME_VERSION` in `config.py`
points to a section here, and every training run's `run_card.md` records it.

## v2: continuous space, 90 degree cone, catch by distance and facing, anti-farming rewards

The current version. Same kind of game as v1 (two-team zero-sum POSG, 2 seekers v 1 hider by default),
but played in continuous space. The grid version is kept in git under the tag `v1-grid`.

### State
- every agent's position (x, y), real numbers, and heading (the direction it looks, any angle)
- which hiders are still alive, and the step count t
- The map is fixed background: every `#` cell of the map file is a solid 1 x 1 square.
  1 unit = 1 map cell, 1 step = about 1 second.

### Actions
Every step each agent picks two things at once (two separate choices, from two heads of its network):
- **move:** stay, forward, back, strafe left, strafe right (relative to where it looks), 1 unit per step for all of them
- **turn:** one of -90, -30, -10, -3, 0, +3, +10, +30, +90 degrees. The biggest turn is 90 degrees, so turning round takes
  2 steps: that is what makes flanking and attacking from behind worth something.
- So 5 x 9 = 45 possible (move, turn) pairs per agent.

### Transitions
- Each agent turns first, then moves. Agents are circles of radius 0.3. A move is done in 4 small pieces; after each
  piece the agent is pushed out of any wall it overlaps, so it slides along walls instead of stopping dead, and can never
  pass through a wall or a corner gap.
- Agents do not block each other (they can overlap), as in the grid version.
- Seekers are frozen (no moving, no turning) during the hiding phase. Caught hiders leave the game.

### Vision
- Each agent sees a **90 degree cone** centred on its heading, with no range limit, blocked by walls.
- **Enemies:** an enemy is seen if the straight line between the two centres touches no wall and the enemy's centre is
  inside the viewer's cone. This is checked exactly, so nobody can slip between rays. A line passing exactly through the
  point where two walls touch counts as blocked (no seeing through corner gaps).
- **Sight is no longer two-way.** A seeker can see a hider's back while the hider cannot see it. This is intended
  (it is how an FPS works), but it means a hider can lose reward because of a seeker it cannot observe.
- **Rays:** 31 rays, one every 3 degrees across the cone, each stopping at the first wall or the first body it hits.
  They describe the layout around the agent (they are what the network "looks at").

### Catch
A hider is caught when a seeker (not frozen) has it in sight, their centres are at most 0.8 apart, and the hider is
within 30 degrees either side of where the seeker looks. So the seeker has to actually face the hider.

### Rewards (shared by the whole team)
Written from the seekers' side; hiders always get exactly minus this (zero-sum).

| Event | Seekers |
|---|---|
| hiding phase step | 0 |
| a hider is in sight | +0.05 x closeness (1 when touching, falling evenly to 0 at 10 units; the best pair counts) |
| no hider in sight | -0.05 |
| each catch | +10 |
| all hiders caught | +10, plus up to +10 for time left (10 x steps left / hunting steps) |
| hiders survive to the time limit | -10 |

- **Why (fixes the farming exploit):** in v1 seekers got +1 every step they could see a hider, and catching ended the round,
  so following a hider for about 26 steps or more paid more than catching it. Now the per-step rewards are tiny next to a
  catch: watching a hider 3 units away for a whole round is worth about -7 to the seekers, catching it at once about +30
  (checked in `test_env.py`).
- Closeness only counts while a hider is seen. Hiders are not rewarded for distance while unseen: they could not observe
  it, and it would teach running to corners instead of hiding.

### Observations
Each agent gets four things:
1. **rays** (31 x 4): for each ray, its length (divided by 20) and whether it hit a wall, a teammate or an enemy
2. **enemies** (5 x 4): enemies it can see right now, nearest first: a "seen" flag, the direction relative to where it
   looks (as cos and sin) and the distance (divided by 20). Empty slots are 0. 5 slots so team sizes can grow later.
3. **6 numbers:** its position (scaled to 0 to 1), its heading (cos and sin), the fraction of the round gone, hiding phase
4. **minimap** (5 x 12 x 12), as in v1.2: walls, me, teammates, team "searched recently" and my own "last seen enemy" memory.
   "Searched" now means inside one of the team's cones (map cells the rays pass over), fading to 0 over 10 steps.

The v1 11 x 11 close-up is gone: the rays replace it.

### End of round and discount
Unchanged: the round ends when all hiders are caught or after 100 steps (15 of them the hiding phase). Discount 0.99.

## v1: 360 degree vision, adjacent catch, team rewards with win/loss bonus

(The last grid version is v1.2, see the changelog: same rules and rewards, richer observations.)

### Agents
- m seekers and n hiders, so m + n agents, in two teams. Training starts with 1v1.
- **Within a team: cooperative.** Teammates share one reward and one network (parameter sharing).
- **Between teams: zero-sum.** Every step, the seekers' reward is exactly minus the hiders'.
- So the game is a **two-team zero-sum POSG**. Each team on its own is a cooperative problem
  (a Dec-POMDP); the two teams together are a zero-sum stochastic game.

### State
Everything the game engine needs to compute the next moment exactly:
- position of every seeker and every hider on the grid
- which hiders are still alive
- the step count t (the hiding phase and the time limit depend on it)

The map (walls, spawns) never changes, so it is fixed background rather than part of the state.

**Agents' memory is not part of the state.** Deleting every memory would not change how the game
evolves, only how well agents decide. Memory is each agent's **belief**: its own summary of past
observations, used to make up for partial observability. It is stored in `env.py` only for convenience.

### Actions
- Each agent: {stay, up, down, left, right}, 5 actions.
- Joint action = one action per agent, all chosen at the same time. 1v1: 5 x 5 = 25 joint actions; 2v2: 5^4 = 625.
- Seekers' actions are ignored during the hiding phase; caught hiders no longer act.

### Transitions
- Deterministic given the joint action: everyone moves at once, walls block moves, agents may share cells.
- Then catches are checked: a hider is caught if a seeker is on the same or an adjacent cell (not during the hiding phase).
- The only randomness is where agents spawn at the start of a round. From each agent's point of view the game
  still looks random, because the other team's choices are unknown.

### Rewards (shared by the whole team)
For hiders each step (seekers get exactly minus this):

| Event | Hiders | Seekers |
|---|---|---|
| hiding phase step | 0 | 0 |
| no hider visible to any seeker | +1 | -1 |
| any hider visible (or caught this step) | -1 | +1 |
| each catch | -5 | +5 |
| round ends, hiders survived (time limit) | +10 | -10 |
| round ends, all hiders caught | -10 | +10 |

### Observations
Each agent sees an 11 x 11 square centred on itself (v1 used 9 x 9, see changelog), with 5 layers:
1. walls (outside the map counts as wall)
2. teammates
3. enemies it can see right now (within range 5, line of sight not blocked by walls)
4. memory: where it last saw enemies, fading by x0.9 per step, cleared when it looks and they are gone
5. the cells it can currently see

Plus 4 numbers: its row and column (scaled to 0 to 1), the fraction of the round gone, and whether it is still the hiding phase.

From v1.2, a 6th close-up layer and a minimap are added (see changelog).

### End of round and discount
- The round ends when all hiders are caught (seekers win) or after 100 steps (hiders win).
- Discount 0.99: a reward 100 steps away is worth about 0.99^100 = 0.37 of its face value now.
  Big enough that the win/loss bonus at the end still matters from the start of the round.

### Notes
- **Markov property:** one observation is not enough to act well, because enemies out of sight still exist.
  The memory layer is a hand-made fix; a recurrent network (GRU) is the learned alternative, planned later.
- **Possible reward exploits to watch for:** hiders camping in a corner the seekers rarely check;
  seekers standing still in a spot with a long view rather than searching.

## Changelog
(each new version gets a section here: what changed in which piece, and why)

### v1.1: observation window 9 x 9 -> 11 x 11
- **What changed:** only the observations. Rules, rewards and transitions are identical to v1.
- **Why:** vision range is 5 cells, but a 9 x 9 window only reaches 4 cells each way. So a seeker 5 cells away
  in a straight line could see the hider (and the hider got -1) while the seeker was outside the hider's own
  observation. Measured with random play: 5% of "seen" steps had this blind spot. The window now always covers
  the vision range, and `env.py` refuses to start if they ever disagree again.
- **Found from:** 1v1 training where hiders did not run away when spotted.

### v1.2: shared team search memory and a minimap (first labelled "v2" here, renamed so v2 can be continuous space)
- **What changed:** only the observations. Rules, rewards and transitions are identical to v1.1.
- **Close-up view gets a 6th layer, "searched recently":** every cell anyone on my team can see right now is set
  to 1, and fades by 0.1 per step, reaching 0 after 10 steps unseen (`SEARCH_MEMORY_STEPS`; think of a step as
  about a second). It is **shared by the team**, like callouts or a minimap in an FPS: a seeker sees what its
  teammate already checked. Both teams have their own, so observations stay the same shape for both.
- **New minimap:** the whole map shrunk to 12 x 12 (`MINIMAP_SIZE`), the same size on every map, with 5 layers:
  walls, me, teammates, how recently my team searched each area, and my own "last seen enemy" memory.
- **Why:** with only an 11 x 11 close-up, an agent cannot tell which parts of the map it has already searched,
  so seekers can walk in circles. This matters most on the big map, where the close-up covers a small part of it.
  The memory is hand-made (a fixed rule), the simplest kind of belief. A learned memory (GRU) is the alternative.
- **What stays individual:** each agent's "last seen enemy" memory is still its own, not shared.
- **Also decided:** the seekers' -1 per step while no hider is visible stays. It depends only on whether a hider
  is visible, not on movement, so there is no way to dodge it except finding the hider.

### v2: continuous space with cone vision (replaces the grid)
- **What changed:** everything about the world. Continuous positions and headings, a 90 degree cone instead of 360
  degree vision with range 5, exact enemy sight, catching by distance and facing, two-part actions (move and turn),
  new rewards, rays instead of the close-up grid. The full definition is in the v2 section above.
- **Why:** the grid was not FPS-like: all-round vision with a made-up range limit, no facing (so no flanking),
  sight leaking through corner gaps, blocky movement. Continuous space is also the step towards continuous time.
  The rewards change fixes the farming exploit (see the v2 rewards).
- **Kept:** 2v1 on the tiny map, hiding phase, team search memory and minimap, opponent pool, bots (rewritten to steer).
- **Benchmark bots in v2:** the chaser now wanders to random spots instead of moving randomly, and the fleeing hider waits
  still while looking around, then backs away from a seeker while facing it. Reference numbers, 2v1 tiny, 300 games:
  random v random 20%, chaser v random hider 89%, chaser v fleeing bot 74%, random seekers v fleeing bot 4%.
  These are not comparable with the grid bot numbers.

### Not game changes (training only, recorded in each run card instead)
- Opponent pool: some training games are played against frozen past versions of the other team.
  This changes how agents learn, not what the game is, so it does not get a game version.
- Benchmark bots (`bots.py`): a scripted chaser and a scripted fleeing hider, used only to measure progress.
  In 1v1 on `tiny.txt` the fleeing bot is never caught by the chaser, which shows that on this map, with equal
  speeds, a hider that runs away correctly can always survive.

### Experiment notes
- **1v1, tiny, v1 (plain self-play, 2M steps):** hiders learned to run (random seekers won only 1.2% against them),
  seekers learned some searching (50% vs random hiders, random baseline 33%), but trained vs trained ended 0%:
  hiders always escape, which matches the bot result above. Once seekers could never win, every action gave
  the same -1, so they had nothing to learn from and stopped searching. Self-play win rate swung back and forth
  the whole run (cycling).
- **Bot check for team sizes (seekers win, chaser bots vs fleeing bots):** 1v1 0%, 2v1 38%, 3v1 65%, 2v2 22%.
  2v1 is the first balanced setting, and the bots don't even coordinate, so trained seekers that work together
  should do better. Next: 2v1 with the opponent pool, on v1.2.
