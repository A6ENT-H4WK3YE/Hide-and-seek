# Game formulation

The game is a **partially observable stochastic game (POSG)**: several agents act at the same
time, the game moves on according to their joint action, and each agent only sees part of what
is going on. This file says exactly which game each version is. `GAME_VERSION` in `config.py`
points to a section here, and every training run's `run_card.md` records it.

## v4.1: taking players out by keeping them in your sights, on a balanced two-site map

The current version: v4 (below) with two changes, both from Moeez's feedback on the first v4 design.
- **Keep them in your sights instead of tags.** A player is out once an enemy has kept them in sight, within 30 degrees
  of where it looks (roughly "crosshair on them"), for **10 steps in a row** (about 10 seconds). Any step without them
  in view restarts that count. Whoever starts watching first finishes first, so holding an angle beats running in blind;
  players finishing on the same step (for example two players watching each other from the same moment) are settled by
  a coin flip, and a player taken out earlier in that step does not get its own takedown. Each watcher counts on its own.
  No takedowns during setup. This replaces the tag rules (two different enemies within 8 steps) and their edge cases,
  and rewards awareness of your surroundings. Tag reward removed; takedown +10 stays.
  Why 30 degrees and not the whole 90 degree cone: it makes aim and tracking matter, not only position.
- **New map `maps/two_sites.txt`**, two sites instead of three (two defenders cannot cover three sites), mirror
  symmetric so A and B are equally fair, and no line of sight from one spawn to the other (0 of 972 spawn cell pairs;
  the old layout had a straight mid corridor between the spawns, and vision has no range limit). Mid is reached through
  side doors from each spawn; each site has a long outer lane, a lower-mid route and a mid door, and mid links the two
  sites. Walking distance to a site: defenders 12, attackers 22; A to B rotation 20.
- **Spike 40 -> 50 steps**, so a retake from the other site (20 steps away) plus a defuse (7) is possible.
- **Bots:** defenders now hold one site together; a bot that sees an enemy stops and keeps its aim on it; a bot whose
  team only knows where an enemy is walks towards it quietly. Bots v bots on the new map: attackers win 51%, spike
  planted 71%, defused 31%, takedowns per round 0.43 by attackers and 1.66 by defenders (holding an angle wins fights,
  attackers win through the objective).
- The first v4 run (three sites, tags) was stopped at 300k steps after Moeez's feedback; attackers had never planted.
- **Walking into walls costs 0.02** (Moeez): a player that tries to move but gets less than half way (blocked by a wall)
  pays this alone; the other team does not gain it, so it is the one part of the rewards that is not zero-sum. Far too
  small to change who wins (a round is worth 20), it only teaches clean movement. Logged as `wall_bump_rate` (random play
  bumps on about 5% of moves).

## v4: attack and defend (Valorant / CS style), with tags instead of guns

Superseded by v4.1 above (tags replaced, new map); everything else below still holds. A new game built on everything v1 to v3.5 developed (continuous movement, 90 degree vision cones,
rays, sound through walls, walk / run, learned memory, the opponent pool, parallel training). The hide-and-seek game
(v1 to v3.5) is kept in git under the tag `v3.5-hide-and-seek`; its sections and results are further down.

**Why a new game (Moeez):** the end goal is an FPS like Valorant: teams going to a site to do an objective while the
other team defends it. Hide and seek had one side that only runs, which is not that game, and it kept running into the
same walls: whichever side could not win stopped learning, and the seekers never needed teamwork. Moeez proposed tags
that need two teammates, and a symmetric 2v2 where a player tagged twice is out and a lone survivor cannot take anyone
out. He asked for a complete, reasoned design in one go rather than stage by stage; the choices below are Claude's.

### Teams and round
- **Attackers** (2, spawn at the bottom, `s` cells) and **defenders** (2, spawn at the top next to the sites, `h` cells).
  Each team has its own network (shared by its players), both with a learned memory (GRU).
- 250 steps (1 step is about a second). For the first 30 (**setup**) attackers are frozen and nobody can tag, so
  defenders can take positions (like Valorant's buy phase).

### Movement, vision, sound
As in v3.5: circles of radius 0.3 sliding along square walls; move (stay, forward, back, strafe left / right), turn
(-90 to +90 degrees) and keep / switch walk-run mode (walk 0.5 units per step and silent, run 1 and noisy; every round
starts walking); a 90 degree vision cone with no range limit, blocked by walls, sight one-way. Running, planting and
defusing make noise, heard by enemies within a quarter of the map's longer side (11.25 cells), through walls.

### Tags and eliminations (the "shot")
- A **tag** needs the enemy in sight, within **3 units** and within **20 degrees** of where you look: like a short-range
  shot that needs aim. Each player tags at most one enemy per step (the nearest it can).
- **One tag only marks a player.** A player is **out** once tagged by **two different enemies within 8 steps**.
  - Why: taking someone out needs two teammates working together (one keeps a tail and calls it in, the other closes
    in), which is Moeez's idea, and it rewards crossfires and trading, as in real FPS teamplay.
  - It also gives his "lone survivor" rule for free: a player whose teammate is out cannot take anyone out alone
    (unless the teammate's tag is still inside the window: a "trade"), so the right play is to survive or play the
    objective. In a 1v1 the rule is capped at the enemy team's size, so one tag is enough.

### Objective: the spike
- An attacker **standing still on a site cell for 5 steps** in a row plants the spike there (moving resets it).
- Then the round timer stops and the spike counts down **40 steps**: attackers win when it explodes. Defenders win by
  **defusing**: standing still within 1 unit of the spike for **7 steps** in a row. Everyone knows where a planted spike
  is and how long is left (as in Valorant).
- After the plant, taking out every attacker does not win on its own: the spike still has to be defused.
- Why the spike, not "plant = win": with 3 sites and 2 defenders, an instant win made it a race to the empty site
  (scripted bots: attackers 78%). Retakes bring it back to a fight (bots: 64%) and are where teamwork matters most.

### Round end
| How | Winner |
|---|---|
| the spike explodes, or every defender is out | attackers |
| the spike is defused, every attacker is out before the plant, or time runs out before the plant | defenders |
| both teams out on the same step (before a plant) | draw |

### Rewards (attackers; defenders get exactly minus this, shared by the team)
| Event | Attackers |
|---|---|
| each fresh tag on an enemy (re-tagging inside the window pays nothing) | +2 |
| each enemy taken out | +10 |
| spike planted | +5 |
| round won / lost | +20 / -20 |
| site shaping (potential-based) | +0.997 x new potential - old potential |

- Only real game events count (tags are like damage dealt, the plant is a milestone).
- **Site shaping:** the potential is minus the walking distance from the closest attacker to a site, divided by the
  longest such distance on the map (0 once the spike is down). Walking towards a site pays a little and walking away
  pays it back, so it cannot be farmed; shaping of this form is proven not to change which strategies are best (also
  in zero-sum games), it only helps attackers find the sites early in training, when random attackers never plant.
- No reward for being seen or unseen, for distance, or for exploring: everything else has to come from winning.

### Observations (each player)
- **rays** (31 x 4): distance, and wall / teammate / enemy
- **enemies** (5 x 9): every enemy anyone on my team sees or hears right now (**team callouts**), nearest first: direction
  relative to where I look, distance, I see it, a teammate sees it, only heard, tagged by my team recently (time left in
  the window), planting
- **mates** (4 x 7): my teammates: still in, direction, distance, tagged recently, planting, sees an enemy
- **vec** (20): position, heading, time, setup phase, running, tagged recently, my plant progress, the attackers' best
  plant progress, on a site, attacker or not, teammates and enemies still in, and the spike: down, time left,
  direction, distance, defuse progress
- **minimap** (6 x 12 x 12): walls, sites, me, teammates, where my team searched recently (fading over 60 steps) and where
  my team last saw or heard enemies (fading over about 30 steps)
- Slots are sized for teams of up to 5, so the same networks can grow towards 5v5.

### Training
- PPO, both teams learning from scratch, discount 0.997 (a win at the end still counts about half at the start).
- Both teams have memory; learning replays it in pieces of 32 steps, each starting from the exact memory it had in play.
- Opponents: 20% of games one team (picked at random) is played by the **scripted bots** (`bots.py`); of the rest, half
  have one team played by a **past version** from the opponent pool (spanning the whole run). Only current selves learn.
- Bots: attackers pick one site and go together, fight anything the team knows about, plant when the site is clear and
  guard the spike; defenders each hold a random site, go for known attackers together and retake a planted spike.
  Bots v bots: attackers win about 64%; random players never plant.

## v3: walk or run, with footsteps you can hear

The current version is v3.2: walk / run is a mode you switch (v3.1), and sound goes through walls with a range that
scales with the map (v3.2). See the changelog; the hearing rule below is the original v3 one. Everything in v2.1 (below) stays, plus sound. Default map is now `three_sites.txt`
(250-step rounds, 30-step hiding phase).

### Actions
A third choice every step, next to move and turn: **walk** or **run** (like holding Shift in CS).
- **Run:** full speed (1 unit per step), and if the agent actually moves, it makes noise this step.
- **Walk:** half speed (0.5 units per step), silent. Turning or standing still is always silent.
- So 5 x 9 x 2 = 90 possible actions per agent. Seekers frozen in the hiding phase make no noise.

### Hearing
- An agent hears an **enemy** that ran this step if the walking distance between their map cells is at most
  **6 cells**. Walking distance (the shortest path along the floor) means sound goes round corners but not
  through walls: two agents either side of a wall can be 2 apart in a straight line and still not hear each other.
- Teammates' footsteps are not reported (only enemies matter).
- Works the same both ways: hiders hear running seekers, seekers hear running hiders.

### Observations
A fifth part, **heard** (5 x 4): enemies heard right now, nearest first: a flag, the direction to them relative to
where I look (cos, sin) and the walking distance (divided by 20). A heard enemy is also marked in my own
"last seen enemy" memory on the minimap, the same as a seen one.

### Why
In v2.1 hiders only knew a seeker was there if it was inside their 90 degree cone, so anyone coming from behind was
invisible until too late. Footsteps give awareness all the way round, without a reward to exploit. It also creates a
trade-off for both teams: run and be heard, or walk quietly at half speed.

## v2: continuous space, 90 degree cone, catch by distance and facing, anti-farming rewards

Same kind of game as v1 (two-team zero-sum POSG, 2 seekers v 1 hider by default),
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
| a hider is in sight | +0.2 x closeness (1 when touching, falling evenly to 0 at 10 units; the best pair counts). v2 used 0.05, see v2.1 |
| no hider in sight | -0.05 |
| each catch | +10 |
| all hiders caught | +10, plus up to +10 for time left (10 x steps left / hunting steps) |
| hiders survive to the time limit | -10 |

- **Why (fixes the farming exploit):** in v1 seekers got +1 every step they could see a hider, and catching ended the round,
  so following a hider for about 26 steps or more paid more than catching it. Now the per-step rewards are small next to a
  catch: in v2.1, watching a hider from 0.9 away (just out of reach) for a whole round is worth about +5.5 to the seekers,
  catching it at once about +30 (checked in `test_env.py`).
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

### v2.1: reward for keeping a hider in sight 0.05 -> 0.2 per step
- **What changed:** only `REWARD_SEEN`. Everything else is identical to v2.
- **Why:** in the first v2 run seekers learned to strafe sideways while spinning (86% of moves strafe right, the most
  common turn +90), sweeping the room like a lighthouse. Even with a hider in sight they never walked forward and lost
  it half the time by the next step. With only 0.05 a step for keeping a hider in view, locking on was hardly worth
  learning, and a spinning seeker near a hider lands the catch anyway once the hider falls in its 30 degree window.
  At 0.2, keeping the hider in view is worth learning, and hiders (who get the opposite) get a real reason to break
  line of sight.
- **Still no farming:** watching from 0.9 away for a whole round is worth about +5.5, catching about +30.

### v3: walk or run, with footsteps (see the v3 section above)
- **What changed:** actions get a third part (walk or run), running makes noise heard by enemies within 6 cells of walking
  distance, and observations get a "heard" part. Rules for vision, catching and rewards are identical to v2.1.
- **Why:** hiders could not tell a seeker was behind them. Walking speed and noise go in together: a slower walk without
  noise would be pointless, and noise without a quiet option gives no choice.
- **Bots:** they now also react to what they hear. The chaser always runs; the fleeing hider waits silently and runs
  when it flees. Reference numbers on three_sites (2v1): random v random 0%, chaser v random hider 97% (v2.1: 70%),
  chaser v fleeing bot 82% (v2.1: 51%): the fleeing bot runs, so the chaser hears it.
- **Watch out (not changed yet):** with 250-step rounds the anti-farming margin is almost gone. Watching a hider from 0.9
  away for a whole round pays +30.0, catching it at once +30.1 (undiscounted). With the 0.99 discount the agents actually
  compare about +15 against +30, so catching still wins, but a fix is due (for example scaling the seen reward by round length).

### v3.1: walk / run becomes a mode you switch, starting in walk (plus an opponent pool covering the whole run)
- **What changed in the game:** the third action part is now "keep my mode" or "switch walk / run" instead of picking walk
  or run fresh every step. Every round starts in walk mode, frozen seekers cannot switch, and the agent's numbers ("vec",
  now 7) include whether it is running. Everything else is identical to v3.
- **Why:** in v3 hiders ran 79% of the time, and the same trained hiders forced to walk were caught far less (seekers'
  self-play wins 77% -> 48%). Being quiet only pays if you keep it up for many steps, which is hard to stumble on when
  walk or run is a fresh coin flip every step. With a mode that stays on, one decision keeps you quiet, and quiet is the default.
- **Training change made at the same time (Moeez asked for both together, so the result cannot be split between them):**
  the opponent pool now spans the whole run. When it is full (20 snapshots), every second snapshot is dropped and snapshots
  are taken half as often, so it always runs evenly from the first random network to the latest. Before, it only held the
  last ~400k steps, so after the seekers got good every frozen seeker the hiders met was already strong.
- **Bots:** they switch whenever their mode is not the one they want (chaser always runs, fleeing hider walks while
  waiting and runs while fleeing). Reference numbers on three_sites 2v1: chaser v random hider 97%, chaser v fleeing bot 78%.

### v3.2: sound goes through walls, range scaled to the map
- **What changed:** only hearing. A running enemy is heard if the **straight-line** distance is at most
  `HEARING_FRACTION` (0.25) x the map's longer side: 11.25 on three_sites (45 x 31), 2.75 on tiny (11 x 11).
  Walls do not block sound. The "heard" observation now gives the straight-line distance. Everything else is v3.1.
- **Why (Moeez):** with 6 cells of walking distance, sound almost never reached anyone on the big map. A hider waiting
  at its spawn only heard a seeker once it was in site B (6 away); a seeker running in site A or C is about 10 away in a
  straight line but 15 to walk, so it was never heard. Hiders had no idea where seekers were, and seekers could not
  hear hiders either. Now a seeker running at the map centre or in A / C is heard from the hider spawn.
- **Bots:** unchanged code, but they hear much further: on three_sites 2v1 the chaser now catches a random hider 100%
  and the fleeing bot 88% (v3.1: 97% and 78%), because the fleeing bot runs when it flees.
- **Run before this change:** the v3.1 run was stopped at 3.4M of 5M steps (`runs/v3.1_three_sites_2v1_20261021_161950`);
  seekers were winning only about 30% of self-play games by then (v3: 85%), so the walk toggle and the longer pool
  probably helped hiders. Not evaluated yet.

### v3.3: hiders pay a small cost for making noise (trained alone, against frozen v3.2 seekers)
- **What changed in the game:** hiders lose 0.02 (`HIDER_NOISE_COST`) every step a hider runs and moves. Seekers' rewards
  are unchanged, so the game is **no longer exactly zero-sum**: this cost is a hint for the hiders, not a rule of the
  match. Everything else is v3.2.
- **Training changes made at the same time (Moeez agreed to all three together, so the result cannot be split):**
  - **only the hiders learn** (`TRAIN_TEAMS = ["hiders"]`). The seekers are frozen: the final v3.2 seekers play in normal
    games, and 17 of the v3.2 run's saved checkpoints (from a random network to the final one) are their opponent pool,
    so new hiders also meet seekers they can beat.
  - the walk / run switch starts biased towards "keep" (`SWITCH_KEEP_PROB = 0.95`: an untrained network switches 5% of
    the time instead of 50%).
  - the hiders start from scratch.
- **Why (Moeez):** fix the hiders first. In v3.2 they pressed switch on 61% of steps, so they ran about half the time and
  the seekers heard them through walls (seekers won 98%). Retraining both teams would just let the seekers win again;
  a quiet, balanced hider comes first, then both teams train together again, then team spotting.
- **Not chosen:** a reward for being far from seekers (hiders cannot observe distance when they neither see nor hear
  a seeker, and it would teach running to corners instead of hiding).

- **Hiders-only run stopped by Moeez at 1.85M of 5M steps** (`runs/v3.3_three_sites_2v1_20261021_185327`): against the
  final v3.2 seekers the new hiders already survived almost every round (seekers won 1.5%, the hider was spotted on 0.6%
  of steps). Moeez: the seekers must keep learning too, from their trained point, so they adapt to the new hiders.
- **Next run (same v3.3 rules): both teams learn again**, seekers starting from the final v3.2 network and hiders from
  the v3.3 network above (`START_FROM` in config). At the start the seekers win only about 2% of self-play.

### v3.4: the team's "searched recently" memory lasts about a minute (60 steps instead of 10)
- **What changed:** only `SEARCH_MEMORY_STEPS`, 10 -> 60. Each map cell inside a teammate's cone is marked 1 and now
  fades by 1/60 per step instead of 1/10, so the minimap shows what the team searched in the last minute.
  No reward changes. Everything else is v3.3.
- **Why (Moeez):** seekers left dead zones (in v3.2 they barely visited site A) and seemed not to know where they had
  already been. They had the memory, but it vanished after 10 steps, while a round lasts 250: by the time a seeker
  crossed the map it had forgotten what it searched. This gives them the information instead of paying them to explore.
- **Considered and kept in reserve:** an exploration reward. A plain bonus for new ground would change the best
  strategy (touring the map instead of hunting), which is tweaking rewards for a result. If dead zones remain, the valid
  version is potential-based shaping (reward when team coverage rises, paid back when it fades), which provably does not
  change the best strategy, only how fast it is found.
- **Run:** both teams learn, seekers starting from the final v3.2 network and hiders from the v3.3 network (`START_FROM`).
  The previous run with these starting points (`runs/v3.3_both_*`) was stopped at 213k steps to make this change; seekers
  had gone from 2% to 10% of self-play by then.

### v3.5: hiders get a learned memory; no noise cost; win / loss reward doubled
Several changes in one run, at Moeez's request; the behaviour measurements separate them where possible
(running away when a seeker is heard: noise cost; varying the hiding spot: memory and the bigger win / loss reward).
- **Hider memory (model, not the game):** a GRU carried from step to step within a round and wiped at the start of the
  next, so a hider can remember where it has been, what it heard and which plan it chose. Its output joins the normal
  features through a layer that starts at zero, so the hiders begin exactly as the v3.4 hiders (checked: identical
  action probabilities). Training replays each hider's steps in order (checked: replay matches play to 3e-6 across
  round ends). Seekers keep their network without memory.
- **No noise cost** (`HIDER_NOISE_COST` 0.02 -> 0): hiders now walk 100% of the time, and the cost stopped them ever
  running away (when a v3.4 hider heard a seeker it never ran, and the seeker closed in by 0.5 per step). The game is
  zero-sum again.
- **Win / loss reward doubled** (`REWARD_WIN` 10 -> 20, `REWARD_LOSS` -10 -> -20): the round result counts for more,
  for hiders (surviving) and for seekers (searching further). Also restores the anti-farming margin: watching a hider
  from 0.9 away for a whole round is worth about +20, catching it about +40.
- **Why (Moeez):** in v3.4 the hiders were silent but all went to site A, and the seekers learned that. He preferred
  letting hiders learn to vary their spot through memory over handing them a random "plan" number.
- **Run:** both teams learn, seekers from the v3.4 seekers, hiders from the v3.4 hiders with the memory added.
- **GIFs (not the game):** sound is drawn (a dotted circle shows how far a running agent's footsteps carry, a yellow
  dotted line joins each listener to the footsteps it hears, "HEARD" in the title), and GIFs play at half speed (3 fps).

### Not game changes (training only, recorded in each run card instead)
- **Escape bot and a league of opponents (2026-10-22, from the v3.5 networks):** a new scripted hider, the escape bot,
  cheats on purpose (it always knows where both seekers are): it heads for the spot furthest from the seekers among
  those it can reach first, faces the nearest seeker, walks while out of earshot and runs once a seeker could hear or
  see it. As a test it showed the trained seekers were not good in general: chaser bots catch it 71%, v3.5 seekers 27%,
  v3.4 seekers 6%. Since then it also plays the hiders in a quarter of training games (`ESCAPE_BOT_PROB`); seekers
  learn from those games, hiders do not. So seekers train against current hiders, past hiders and the escape bot.
- **Discount 0.99 -> 0.997** (from the same run): with 250-step rounds and 0.99, a hider's win at the end was worth only
  2.2 of its 20 points at the start of the hunt (with 0.997, 10.3), so hiders could not feel the value of plans that
  pay off late (hiding, slipping away from footsteps). Seekers' rewards (catches) come sooner, which is why they always
  learned faster. Moeez's idea of rewarding hiders for spotting seekers unseen was not taken: it would be reward tweaking.

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
- **2v1, tiny, v2 (continuous, opponent pool, 2M steps):** seekers learned (65% against random hiders, random seekers 20%;
  21% against the fleeing bot, random seekers 4%) and split up to search. Hiders did not: the chaser bot catches them 91%
  of the time, the same as a random hider (88%). Self-play seeker win rate rose from about 20% to about 45%.
  Next: a longer run to see whether hiders start learning once the seekers settle.
- **2v1, tiny, v2.1 (seen reward 0.2, 2M steps):** the seekers stopped spinning and lock on: with a hider in sight they walk
  forward 74% of the time (v2: 0%). 96% against random hiders, 67% against the fleeing bot (chaser bots 79%), self-play 86%.
  While searching they walk backwards most of the time, since backpedalling is as fast as walking forward.
  Hiders are still no better than random against the chaser bot (91% caught, random hider 88%).
- **2v1, three_sites, v2.1 (5M steps, 250-step rounds, 30-step hiding phase):** both teams learned. Trained hiders are caught by the
  chaser bot 54% of the time (random hider 70%, fleeing bot 51%); trained seekers beat random hiders 77% (random seekers 1%, chaser
  bots 70%) and the fleeing bot 37% (chaser bots 51%); self-play 37%. Hiders mostly rotate to site A and the corridors next to it,
  rarely to B in the middle; seekers push up the middle, then spread to the sites. On the tiny map a searching seeker finds you
  almost anywhere, which is probably why hiders only learned on the big map.
- **2v1, three_sites, v3 (sound, 5M steps):** seekers learned to use sound and chase (74% against the fleeing bot, v2.1: 37%;
  self-play 85%). Hiders did not learn to be quiet: they run 79% of the time and keep moving even with nobody around, and the
  chaser bot catches them 94% (random hider 96%). Forcing the same trained hiders to walk cuts the seekers' self-play win rate
  from 77% to 48%, so walking would pay; hiders just never found it. Hiders moved from site A (v2.1) to site C.
- **2v1, three_sites, v3.1 (stopped at 3.4M steps, evaluated with its own hearing rule):** the most balanced run so far.
  Seekers win 31% of self-play; the chaser bot catches trained hiders 87% (random hider 96%, fleeing bot 80%); trained
  seekers beat the fleeing bot 41% (chaser bots 80%).
- **2v1, three_sites, v3.2 (sound through walls, 5M steps):** seekers win 98% of self-play, 99% against random hiders and 76%
  against the fleeing bot (chaser bots 91%); the chaser bot catches trained hiders 99% (random hider 100%). Seekers hold
  their walk / run mode (press switch on 13% of steps); hiders do not (61%), so they make noise about half the time and the
  seekers hear them through walls. The toggle did not do its job for hiders because an untrained network presses switch
  half the time; next idea: start the switch head biased towards "keep".
- **2v1, three_sites, v3.4 (both learning, seekers from v3.2, hiders from v3.3, 5M steps):** hiders became fully silent
  (walk 100% of the time) and are the best so far against the chaser bot (67% caught; random hider 100%, fleeing bot 91%),
  but they all go to site A. Seekers learned that within about 400k steps, go straight to A, and win 97% of self-play;
  they never search site C or the lanes, and became worse against hiders that go elsewhere (84% against random hiders,
  v3.2: 99%; 53% against the fleeing bot, v3.2: 76%).
- **2v1, three_sites, v3.5 (both learning from v3.4, hider memory, no noise cost, win / loss 20, 5M steps):** hiders now vary
  their side (per round: A 12%, B 24%, C 64%; v3.4 was A 90%), but mostly pick C, where the seekers catch them 99% of the
  time, while at A they win 88%. Without the noise cost they run again (run mode 59% of hunting steps) and are heard
  through walls; forced to walk, seekers win 79% instead of 90%. Seekers became far more general (89% against the
  fleeing bot, v3.4: 53%) and win 84% of self-play. The chaser bot catches trained hiders 94% (v3.4: 67%).
