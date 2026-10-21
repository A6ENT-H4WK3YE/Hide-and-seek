# Play the training games in parallel worker processes, so every CPU core can step games at once.
# The game (env.py) is plain Python + numpy and only uses one core, and it is most of the training time,
# so this is where the speed-up is. The network stays in the main process.
#
#   games = ParallelGames(map_file, seeds, n_seekers, n_hiders, n_workers)   one game per seed
#   obs = games.reset()                                                       one observation dict per game
#   obs, rewards, dones, infos = games.step(seeker_actions, hider_actions)    one entry per game
#   games.close()
#
# A finished game is reset straight away inside its worker, so the observation returned for it is already the
# start of the next round (exactly what train.py did before). Each game keeps its own random numbers, so results
# are the same as stepping the games one after another in a single process.
#
# Each worker is this file run as its own Python process; messages go back and forth as pickles over its
# stdin / stdout. (Python's multiprocessing would re-run train.py inside every worker on a Mac.)

import os
import sys
import pickle
import subprocess
import numpy as np


class ParallelGames:

    def __init__(self, map_file, seeds, n_seekers, n_hiders, n_workers):
        self.n_seekers, self.n_hiders = n_seekers, n_hiders
        n_workers = max(1, min(n_workers, len(seeds)))
        self.parts = np.array_split(np.arange(len(seeds)), n_workers)   # which games each worker plays
        # one thread per worker: the matrices are tiny, and threads fighting over cores only slow things down
        child_env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        script = os.path.abspath(__file__)
        self.procs = []
        for part in self.parts:
            p = subprocess.Popen([sys.executable, script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=child_env)
            self._send(p, ("init", map_file, [seeds[i] for i in part], n_seekers, n_hiders))
            self.procs.append(p)
        for p in self.procs:
            assert self._recv(p) == "ready", "a game worker failed to start (see the error above)"

    def _send(self, p, msg):
        pickle.dump(msg, p.stdin, protocol=pickle.HIGHEST_PROTOCOL)
        p.stdin.flush()

    def _recv(self, p):
        return pickle.load(p.stdout)

    def reset(self):
        for p in self.procs:
            self._send(p, ("reset",))
        obs = []
        for p in self.procs:
            obs += self._recv(p)
        return obs

    def step(self, seeker_actions, hider_actions):
        # seeker_actions / hider_actions: one row of actions per game. All workers step at the same time.
        for p, part in zip(self.procs, self.parts):
            self._send(p, ("step", seeker_actions[part], hider_actions[part]))
        obs, rewards, dones, infos = [], [], [], []
        for p in self.procs:
            for o, r, d, info in self._recv(p):
                obs.append(o)
                rewards.append(r)
                dones.append(d)
                infos.append(info)
        return obs, rewards, dones, infos

    def close(self):
        for p in self.procs:
            try:
                self._send(p, ("close",))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.wait(timeout=10)


def serve():
    # the worker side: wait for a message, do it, send the answer back
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr          # anything printed must not end up in the message pipe
    from env import HideSeekEnv
    envs, seeds, n_seekers, n_hiders = [], [], 0, 0
    while True:
        msg = pickle.load(inp)
        if msg[0] == "init":
            map_file, seeds, n_seekers, n_hiders = msg[1:]
            envs = [HideSeekEnv(map_file) for _ in seeds]
            reply = "ready"
        elif msg[0] == "reset":
            reply = [env.reset(seed=s, n_seekers=n_seekers, n_hiders=n_hiders) for env, s in zip(envs, seeds)]
        elif msg[0] == "step":
            reply = []
            for env, sa, ha in zip(envs, msg[1], msg[2]):
                o, r, done, info = env.step(sa, ha)
                if done:
                    o = env.reset(n_seekers=n_seekers, n_hiders=n_hiders)
                reply.append((o, r, done, info))
        else:
            break
        pickle.dump(reply, out, protocol=pickle.HIGHEST_PROTOCOL)
        out.flush()


if __name__ == "__main__":
    # workers are started with the project folder as the working directory, so config.py and env.py are found
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    serve()
