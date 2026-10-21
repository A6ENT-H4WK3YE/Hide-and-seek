# The neural network each team uses. One network per team, shared by every agent
# on that team (parameter sharing), so it works for any team size.
#
# Input:  what one agent sees, as a dict of tensors (one row per agent)
#           rays    (N_RAYS, 4)              distance and what each ray hit
#           enemies (MAX_TEAM_SIZE, 4)       enemies in sight: direction and distance
#           heard   (MAX_TEAM_SIZE, 4)       enemies heard running: direction and walking distance
#           vec     (7,)                     position, heading, time, hiding phase, running
#           minimap (N_MINIMAP_CHANNELS, MINIMAP_SIZE, MINIMAP_SIZE)  the whole map, shrunk
# Output: three sets of action scores, for moving (5), turning (9) and keep / switch walk-run mode (2), and a value
#         (1 number: "how good is my situation", used to judge whether an action turned out better than expected)
#
# Each agent picks its move, its turn and whether to switch walk / run, at the same time, from three separate heads.
# The chance of the triple is the three chances multiplied together,
# so log-probabilities (and entropies) simply add up.

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

    def __init__(self):
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
        # four heads: how to move, how to turn, whether to switch walk / run, and the critic's value estimate
        self.move_head = layer_init(nn.Linear(cfg.HIDDEN, cfg.N_MOVES), gain=0.01)
        self.turn_head = layer_init(nn.Linear(cfg.HIDDEN, cfg.N_TURNS), gain=0.01)
        self.speed_head = layer_init(nn.Linear(cfg.HIDDEN, cfg.N_SPEEDS), gain=0.01)
        self.critic = layer_init(nn.Linear(cfg.HIDDEN, 1), gain=1.0)

    def forward(self, obs):
        n = obs["rays"].shape[0]
        flat = torch.cat([obs["rays"].reshape(n, -1), obs["enemies"].reshape(n, -1), obs["heard"].reshape(n, -1),
                          obs["vec"]], dim=1)
        h = self.body(torch.cat([flat, self.mini(obs["minimap"])], dim=1))
        dists = [torch.distributions.Categorical(logits=head(h)) for head in (self.move_head, self.turn_head, self.speed_head)]
        return dists, self.critic(h).squeeze(-1)

    def evaluate(self, obs, actions):
        # log-probability and entropy of given (move, turn, switch) actions, plus values (used in the PPO update)
        dists, value = self(obs)
        logp = sum(d.log_prob(actions[:, k]) for k, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
        return logp, entropy, value

    def act(self, obs, greedy=False):
        # pick (move, turn, switch) for a batch of agents; also return log-probabilities and values for training
        dists, value = self(obs)
        if greedy:
            parts = [d.probs.argmax(dim=1) for d in dists]
        else:
            parts = [d.sample() for d in dists]
        logp = sum(d.log_prob(p) for d, p in zip(dists, parts))
        return torch.stack(parts, dim=1), logp, value


def obs_to_tensors(obs_list, device):
    # list of per-agent observations -> dict of batched tensors
    return {k: torch.as_tensor(np.stack([o[k] for o in obs_list]), device=device) for k in OBS_KEYS}
