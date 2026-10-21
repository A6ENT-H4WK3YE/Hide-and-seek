# The neural network each team uses. One network per team, shared by every agent
# on that team (parameter sharing), so it works for any team size.
#
# Input:  what one agent sees, as a dict of tensors (one row per agent)
#           rays    (N_RAYS, 4)              distance and what each ray hit
#           enemies (MAX_TEAM_SIZE, 4)       enemies in sight: direction and distance
#           heard   (MAX_TEAM_SIZE, 4)       enemies heard running: direction and distance
#           vec     (7,)                     position, heading, time, hiding phase, running
#           minimap (N_MINIMAP_CHANNELS, MINIMAP_SIZE, MINIMAP_SIZE)  the whole map, shrunk
# Output: three sets of action scores, for moving (5), turning (9) and keep / switch walk-run mode (2), and a value
#         (1 number: "how good is my situation", used to judge whether an action turned out better than expected)
#
# Each agent picks its move, its turn and whether to switch walk / run, at the same time, from three separate heads.
# The chance of the triple is the three chances multiplied together,
# so log-probabilities (and entropies) simply add up.
#
# Memory (v3.5, teams in MEMORY_TEAMS): a GRU, a small learned memory carried from one step to the next within a round,
# so an agent can remember where it has been, what it heard and what plan it chose. Its output is added to the normal
# features through a layer that starts at exactly zero, so a network loaded from a run without memory plays exactly
# as before until it learns to use the memory. The memory is wiped at the start of every round.

import numpy as np
import torch
import torch.nn as nn
import config as cfg
from env import N_MINIMAP_CHANNELS, RAY_FEATURES, ENEMY_FEATURES, VEC_DIM

OBS_KEYS = ["rays", "enemies", "heard", "vec", "minimap"]
ACTION_PARTS = 3     # move, turn, switch walk / run


def get_device():
    if cfg.DEVICE != "auto":
        return torch.device(cfg.DEVICE)
    if torch.cuda.is_available():            # NVIDIA, and AMD with ROCm builds of PyTorch
        return torch.device("cuda")
    if torch.backends.mps.is_available():    # Apple Silicon Macs
        return torch.device("mps")
    return torch.device("cpu")


def layer_init(layer, gain=np.sqrt(2)):
    # standard PPO initialisation: orthogonal weights, zero bias
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class TeamNet(nn.Module):

    def __init__(self, memory=False):
        super().__init__()
        m = cfg.MINIMAP_SIZE
        # minimap branch: reads the whole map (where am I, where has my team searched, ...)
        self.mini = nn.Sequential(
            layer_init(nn.Conv2d(N_MINIMAP_CHANNELS, 16, kernel_size=3, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        # everything else is a list of numbers: rays, enemies in sight, enemies heard, position / heading / time
        flat = cfg.N_RAYS * RAY_FEATURES + 2 * cfg.MAX_TEAM_SIZE * ENEMY_FEATURES + VEC_DIM
        # combine it all (two layers, since the raw rays need some work to turn into "a doorway on my left")
        self.body = nn.Sequential(
            layer_init(nn.Linear(flat + 16 * m * m, cfg.HIDDEN)), nn.ReLU(),
            layer_init(nn.Linear(cfg.HIDDEN, cfg.HIDDEN)), nn.ReLU(),
        )
        # optional memory: what the GRU remembers is added to the features through mem_out, which starts at zero
        self.memory = memory
        if memory:
            self.gru = nn.GRUCell(cfg.HIDDEN, cfg.HIDDEN)
            self.mem_out = nn.Linear(cfg.HIDDEN, cfg.HIDDEN)
            nn.init.zeros_(self.mem_out.weight)
            nn.init.zeros_(self.mem_out.bias)
        # four heads: how to move, how to turn, whether to switch walk / run, and the critic's value estimate
        self.move_head = layer_init(nn.Linear(cfg.HIDDEN, cfg.N_MOVES), gain=0.01)
        self.turn_head = layer_init(nn.Linear(cfg.HIDDEN, cfg.N_TURNS), gain=0.01)
        self.speed_head = layer_init(nn.Linear(cfg.HIDDEN, cfg.N_SPEEDS), gain=0.01)
        # start out mostly pressing "keep", so the walk / run mode stays put while the network is still untrained
        with torch.no_grad():
            self.speed_head.bias.copy_(torch.log(torch.tensor([cfg.SWITCH_KEEP_PROB, 1 - cfg.SWITCH_KEEP_PROB])))
        self.critic = layer_init(nn.Linear(cfg.HIDDEN, 1), gain=1.0)

    def initial_state(self, n, device):
        # an empty memory for n agents (None for a network without memory)
        return torch.zeros(n, cfg.HIDDEN, device=device) if self.memory else None

    def encode(self, obs):
        # everything one agent sees right now -> features (no memory involved)
        n = obs["rays"].shape[0]
        flat = torch.cat([obs["rays"].reshape(n, -1), obs["enemies"].reshape(n, -1), obs["heard"].reshape(n, -1),
                          obs["vec"]], dim=1)
        return self.body(torch.cat([flat, self.mini(obs["minimap"])], dim=1))

    def heads(self, x):
        dists = [torch.distributions.Categorical(logits=head(x)) for head in (self.move_head, self.turn_head, self.speed_head)]
        return dists, self.critic(x).squeeze(-1)

    def forward(self, obs, h=None):
        # one step for a batch of agents: action distributions, values, and the updated memory
        x = self.encode(obs)
        if self.memory:
            h = self.gru(x, h)
            x = x + self.mem_out(h)
        dists, value = self.heads(x)
        return dists, value, h

    def evaluate(self, obs, actions):
        # log-probability and entropy of given (move, turn, switch) actions, plus values (PPO update, no memory)
        dists, value, _ = self(obs)
        logp = sum(d.log_prob(actions[:, k]) for k, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
        return logp, entropy, value

    def evaluate_sequence(self, obs, actions, h0, starts):
        # The same, for whole sequences of steps (PPO update, with memory). obs[k]: (T, B, ...), actions: (T, B, 3),
        # h0: the memory each sequence started with, starts: (T, B), 1 where a new round began (memory wiped).
        # The memory is replayed step by step in order, so learning can see what remembering an earlier step was worth.
        T, B = actions.shape[:2]
        x = self.encode({k: v.reshape((T * B,) + v.shape[2:]) for k, v in obs.items()}).reshape(T, B, -1)
        h, outs = h0, []
        for t in range(T):
            h = h * (1.0 - starts[t]).unsqueeze(1)
            h = self.gru(x[t], h)
            outs.append(x[t] + self.mem_out(h))
        dists, value = self.heads(torch.stack(outs).reshape(T * B, -1))
        flat_actions = actions.reshape(T * B, -1)
        logp = sum(d.log_prob(flat_actions[:, k]) for k, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
        return logp.reshape(T, B), entropy.reshape(T, B), value.reshape(T, B)

    def act(self, obs, h=None, greedy=False):
        # pick (move, turn, switch) for a batch of agents; also return log-probabilities, values and the new memory
        dists, value, h = self(obs, h)
        if greedy:
            parts = [d.probs.argmax(dim=1) for d in dists]
        else:
            parts = [d.sample() for d in dists]
        logp = sum(d.log_prob(p) for d, p in zip(dists, parts))
        return torch.stack(parts, dim=1), logp, value, h


def make_net(team):
    # a fresh network for a team, with memory if the team is in MEMORY_TEAMS
    return TeamNet(memory=team in cfg.MEMORY_TEAMS)


def load_net(state_dict, device):
    # a network from a saved state; it has memory if the saved state does
    net = TeamNet(memory=any(k.startswith("gru.") for k in state_dict)).to(device)
    net.load_state_dict(state_dict)
    return net


def obs_to_tensors(obs_list, device):
    # list of per-agent observations -> dict of batched tensors
    return {k: torch.as_tensor(np.stack([o[k] for o in obs_list]), device=device) for k in OBS_KEYS}
