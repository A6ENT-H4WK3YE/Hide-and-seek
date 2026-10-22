# Play matches between any mix of trained networks, scripted bots and random players.
# Used by train.py (benchmark curves during training) and evaluate.py (final check).
#
#   play_match(env, bots, {"attackers": "bot", "defenders": "trained"}, nets, games=100, device=device)
#   -> {"attacker_win": 0.42, "length": 131.0, "planted": 0.55, "defused": 0.12, "outs_by_attackers": 0.9, ...}

import numpy as np
import torch
from env import TEAMS
import config as cfg
from render import draw_frame, save_gif
from model import obs_to_tensors


def trained_actions(net, obs_team, device, h=None):
    # actions for one team, and its updated memory (None for a network without memory)
    with torch.no_grad():
        a, _, _, h = net.act(obs_to_tensors(obs_team, device), h)
    return a.cpu().numpy(), h


def random_actions(rng, n):
    # one random (move, turn, switch walk / run) triple per player
    return np.stack([rng.integers(0, cfg.N_MOVES, size=n), rng.integers(0, cfg.N_TURNS, size=n),
                     rng.integers(0, cfg.N_SPEEDS, size=n)], axis=1)


def play_match(env, bots, policies, nets, games, device, seed=10_000, gif_path=None):
    # policies: {"attackers": ..., "defenders": ...}, each one of "trained", "bot", "random"
    rng = np.random.default_rng(seed)
    wins, lengths, planted, defused = 0, [], 0, 0
    outs = {team: 0 for team in TEAMS}
    for k in range(games):
        obs = env.reset(seed=seed + k)
        bots.reset()
        frames = [draw_frame(env)] if (gif_path and k == 0) else None
        done = False
        # trained networks with memory start every game with an empty one
        h = {team: nets[team].initial_state(len(obs[team]), device) if policies[team] == "trained" else None
             for team in TEAMS}
        while not done:
            acts = {}
            for team in TEAMS:
                if policies[team] == "trained":
                    acts[team], h[team] = trained_actions(nets[team], obs[team], device, h[team])
                elif policies[team] == "bot":
                    acts[team] = bots.actions(team)
                else:
                    acts[team] = random_actions(rng, len(obs[team]))
            obs, _, done, info = env.step(acts)
            for team in TEAMS:
                outs[team] += info["outs"][team]
            if frames is not None:
                frames.append(draw_frame(env))
        wins += info["winner"] == "attackers"
        planted += info["planted"]
        defused += info["defused"]
        lengths.append(info["t"])
        if frames is not None:
            save_gif(frames, gif_path)
    return {"attacker_win": wins / games, "length": float(np.mean(lengths)), "planted": planted / games,
            "defused": defused / games, "outs_by_attackers": outs["attackers"] / games,
            "outs_by_defenders": outs["defenders"] / games}
