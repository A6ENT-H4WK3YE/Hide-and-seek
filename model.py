# The neural network each team uses. One network per team, shared by every agent
# on that team (parameter sharing), so it works for any team size.
#
# Input:  what one agent sees, as a dict of tensors (one row per agent)
#           rays    (N_RAYS, 4)              distance and what each ray hit
#           enemies (MAX_TEAM_SIZE, 4)       enemies in sight: direction and distance
#           vec     (6,)                     position, heading, time, hiding phase
#           minimap (N_MINIMAP_CHANNELS, MINIMAP_SIZE, MINIMAP_SIZE)  the whole map, shrunk
# Output: two sets of action scores, one for moving (5) and one for turning (9), and a value
#         (1 number: "how good is my situation", used to judge whether an action turned out better than expected)
#
# Each agent picks its move and its turn at the same time, from two separate heads.
# The chance of the pair is the chance of the move times the chance of the turn,
# so log-probabilities (and entropies) simply add up.

import numpy as np
import torch
import torch.nn as nn
import config as cfg
from env import N_MINIMAP_CHANNELS, RAY_FEATURES, ENEMY_FEATURES, VEC_DIM

OBS_KEYS = ["rays", "enemies", "vec", "minimap"]


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
        # everything else is a list of numbers: rays, enemies in sight, position / heading / time
        flat = cfg.N_RAYS * RAY_FEATURES + cfg.MAX_TEAM_SIZE * ENEMY_FEATURES + VEC_DIM
        # combine it all (two layers, since the raw rays need some work to turn into "a doorway on my left")
        self.body = nn.Sequential(
            layer_init(nn.Linear(flat + 16 * m * m, cfg.HIDDEN)), nn.ReLU(),
            layer_init(nn.Linear(cfg.HIDDEN, cfg.HIDDEN)), nn.ReLU(),
        )
        # three heads: how to move, how to turn, and the critic's value estimate
        self.move_head = layer_init(nn.Linear(cfg.HIDDEN, cfg.N_MOVES), gain=0.01)
        self.turn_head = layer_init(nn.Linear(cfg.HIDDEN, cfg.N_TURNS), gain=0.01)
        self.critic = layer_init(nn.Linear(cfg.HIDDEN, 1), gain=1.0)

    def forward(self, obs):
        n = obs["rays"].shape[0]
        flat = torch.cat([obs["rays"].reshape(n, -1), obs["enemies"].reshape(n, -1), obs["vec"]], dim=1)
        h = self.body(torch.cat([flat, self.mini(obs["minimap"])], dim=1))
        move_dist = torch.distributions.Categorical(logits=self.move_head(h))
        turn_dist = torch.distributions.Categorical(logits=self.turn_head(h))
        return move_dist, turn_dist, self.critic(h).squeeze(-1)

    def evaluate(self, obs, actions):
        # log-probability and entropy of given (move, turn) actions, plus values (used in the PPO update)
        move_dist, turn_dist, value = self(obs)
        logp = move_dist.log_prob(actions[:, 0]) + turn_dist.log_prob(actions[:, 1])
        entropy = move_dist.entropy() + turn_dist.entropy()
        return logp, entropy, value

    def act(self, obs, greedy=False):
        # pick (move, turn) for a batch of agents; also return log-probabilities and values for training
        move_dist, turn_dist, value = self(obs)
        if greedy:
            move, turn = move_dist.probs.argmax(dim=1), turn_dist.probs.argmax(dim=1)
        else:
            move, turn = move_dist.sample(), turn_dist.sample()
        logp = move_dist.log_prob(move) + turn_dist.log_prob(turn)
        return torch.stack([move, turn], dim=1), logp, value


def obs_to_tensors(obs_list, device):
    # list of per-agent observations -> dict of batched tensors
    return {k: torch.as_tensor(np.stack([o[k] for o in obs_list]), device=device) for k in OBS_KEYS}
