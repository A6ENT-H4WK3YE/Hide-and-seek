# Play matches between any mix of trained networks, scripted bots and random players.
# Used by train.py (benchmark curves during training) and evaluate.py (final check).
#
#   play_match(env, bots, {"seekers": "bot", "hiders": "trained"}, nets, games=100, device=device)
#   -> {"seeker_win": 0.12, "length": 96.3, "spotted": 0.04}

import numpy as np
import torch
import config as cfg
from render import draw_frame, save_gif
from model import obs_to_tensors


def trained_actions(net, obs_team, device, h=None):
    # actions for one team, and its updated memory (None for a network without memory)
    with torch.no_grad():
        a, _, _, h = net.act(obs_to_tensors(obs_team, device), h)
    return a.cpu().numpy(), h


def random_actions(rng, n):
    # one random (move, turn, switch walk / run) triple per agent
    return np.stack([rng.integers(0, cfg.N_MOVES, size=n), rng.integers(0, cfg.N_TURNS, size=n),
                     rng.integers(0, cfg.N_SPEEDS, size=n)], axis=1)


def play_match(env, bots, policies, nets, games, device, seed=10_000, gif_path=None):
    # policies: {"seekers": ..., "hiders": ...}, each one of "trained", "bot", "random" (hiders also "escape")
    rng = np.random.default_rng(seed)
    seeker_wins, lengths, spotted, hunting = 0, [], 0, 0
    for k in range(games):
        obs = env.reset(seed=seed + k, n_seekers=cfg.N_SEEKERS, n_hiders=cfg.N_HIDERS)
        bots.reset()
        frames = [draw_frame(env)] if (gif_path and k == 0) else None
        done = False
        # trained networks with memory start every game with an empty one
        h = {team: nets[team].initial_state(len(obs[team]), device) if policies[team] == "trained" else None
             for team in ["seekers", "hiders"]}
        while not done:
            acts = {}
            for team in ["seekers", "hiders"]:
                if policies[team] == "trained":
                    acts[team], h[team] = trained_actions(nets[team], obs[team], device, h[team])
                elif policies[team] == "bot":
                    acts[team] = bots.seeker_actions() if team == "seekers" else bots.hider_actions()
                elif policies[team] == "escape":
                    acts[team] = bots.escape_actions()       # cheating escape hider (hiders only)
                else:
                    acts[team] = random_actions(rng, len(obs[team]))
            obs, _, done, info = env.step(acts["seekers"], acts["hiders"])
            if not info["hiding_phase"]:
                hunting += 1
                spotted += info["any_seen"]
            if frames is not None:
                frames.append(draw_frame(env))
        seeker_wins += info["winner"] == "seekers"
        lengths.append(info["t"])
        if frames is not None:
            save_gif(frames, gif_path)
    return {"seeker_win": seeker_wins / games, "length": float(np.mean(lengths)), "spotted": spotted / max(hunting, 1)}
