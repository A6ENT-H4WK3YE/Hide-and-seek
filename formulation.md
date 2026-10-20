# Game formulation

The game is a **partially observable stochastic game (POSG)**: several agents act at the same
time, the game moves on according to their joint action, and each agent only sees part of what
is going on. This file says exactly which game each version is. `GAME_VERSION` in `config.py`
points to a section here, and every training run's `run_card.md` records it.

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
