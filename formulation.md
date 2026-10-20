# Game formulation

The game is a **partially observable stochastic game (POSG)**: several agents act at the same
time, the game moves on according to their joint action, and each agent only sees part of what
is going on. This file says exactly which game each version is. `GAME_VERSION` in `config.py`
points to a section here, and every training run's `run_card.md` records it.

## v1: 360 degree vision, adjacent catch, team rewards with win/loss bonus

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
Each agent sees a 9 x 9 square centred on itself, with 5 layers:
1. walls (outside the map counts as wall)
2. teammates
3. enemies it can see right now (within range 5, line of sight not blocked by walls)
4. memory: where it last saw enemies, fading by x0.9 per step, cleared when it looks and they are gone
5. the cells it can currently see

Plus 4 numbers: its row and column (scaled to 0 to 1), the fraction of the round gone, and whether it is still the hiding phase.

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
