# The neural network each team uses. One network per team, shared by every agent
# on that team (parameter sharing), so it works for any team size.
#
# Input:  what one agent sees  -> grid (5, 9, 9) and vec (4,)
# Output: action scores (5 numbers, one per action) and a value (1 number:
#         "how good is my situation", used to judge whether an action turned out better than expected)

import numpy as np
import torch
import torch.nn as nn
import config as cfg
from env import N_CHANNELS


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

    def __init__(self, vec_dim=4):
        super().__init__()
        w = cfg.OBS_WINDOW
        # convolution layers read the local picture (walls, enemies, memory, ...)
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(N_CHANNELS, 32, kernel_size=3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        # combine the picture with the extra numbers (position, time, hiding phase)
        self.body = nn.Sequential(
            layer_init(nn.Linear(32 * w * w + vec_dim, cfg.HIDDEN)), nn.ReLU(),
        )
        # two heads: the actor picks actions, the critic estimates value
        self.actor = layer_init(nn.Linear(cfg.HIDDEN, cfg.N_ACTIONS), gain=0.01)
        self.critic = layer_init(nn.Linear(cfg.HIDDEN, 1), gain=1.0)

    def forward(self, grid, vec):
        h = self.body(torch.cat([self.conv(grid), vec], dim=1))
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, grid, vec, greedy=False):
        # pick actions for a batch of agents; also return log-probabilities and values for training
        logits, value = self(grid, vec)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=1) if greedy else dist.sample()
        return action, dist.log_prob(action), value


def obs_to_tensors(obs_list, device):
    # list of per-agent observations -> two batched tensors
    grid = torch.as_tensor(np.stack([o["grid"] for o in obs_list]), device=device)
    vec = torch.as_tensor(np.stack([o["vec"] for o in obs_list]), device=device)
    return grid, vec
