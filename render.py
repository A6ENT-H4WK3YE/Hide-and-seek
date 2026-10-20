# Draw the game as images and save GIFs.
#
#   frame = draw_frame(env)            -> RGB numpy array of the current state
#   save_gif(frames, "runs/demo.gif")  -> save a list of frames as an animated GIF

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import config as cfg

WALL = np.array([43, 47, 58]) / 255
FLOOR = np.array([217, 212, 199]) / 255
SITE = np.array([230, 201, 143]) / 255
SEEKER_SPAWN = np.array([236, 214, 208]) / 255
HIDER_SPAWN = np.array([210, 220, 228]) / 255
SEEKER_VIEW = np.array([220, 70, 60]) / 255
HIDER_VIEW = np.array([60, 120, 220]) / 255
SEEKER_COL = "#c0392b"
HIDER_COL = "#2563b0"
BG = "#14161c"


def base_image(env):
    # the map itself: walls, floor, spawn areas, sites
    img = np.zeros((env.H, env.W, 3))
    img[:] = FLOOR
    img[env.grid == "#"] = WALL
    img[env.grid == "s"] = SEEKER_SPAWN
    img[env.grid == "h"] = HIDER_SPAWN
    for site in "ABC":
        img[env.grid == site] = SITE
    return img


def draw_frame(env, show_vision=True):
    img = base_image(env)

    # shade what each team can currently see
    if show_vision:
        seeker_view = np.zeros((env.H, env.W), dtype=bool)
        for p in env.seekers:
            seeker_view |= env.view_of(p)
        hider_view = np.zeros((env.H, env.W), dtype=bool)
        for j, p in enumerate(env.hiders):
            if env.hider_alive[j]:
                hider_view |= env.view_of(p)
        img[hider_view] = 0.8 * img[hider_view] + 0.2 * HIDER_VIEW
        img[seeker_view] = 0.7 * img[seeker_view] + 0.3 * SEEKER_VIEW

    cell = 0.45 if env.W <= 15 else 0.22
    fig, ax = plt.subplots(figsize=(env.W * cell + 0.6, env.H * cell + 1.0), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.imshow(img, interpolation="nearest")

    # thin grid lines on small maps so cells are easy to count
    if env.W <= 15:
        for k in range(env.W + 1):
            ax.axvline(k - 0.5, color=BG, linewidth=0.4)
        for k in range(env.H + 1):
            ax.axhline(k - 0.5, color=BG, linewidth=0.4)

    # dashed lines: which seeker can see which hider
    for i in range(env.n_seekers):
        for j in range(env.n_hiders):
            if env.seeker_sees[i, j]:
                (y0, x0), (y1, x1) = env.seekers[i], env.hiders[j]
                ax.plot([x0, x1], [y0, y1], "--", color=SEEKER_COL, linewidth=1.2)

    # agents
    size = 260 if env.W <= 15 else 70
    for p in env.seekers:
        ax.scatter(p[1], p[0], s=size, color=SEEKER_COL, edgecolors="white", linewidths=1.2, zorder=3)
    for j, p in enumerate(env.hiders):
        if env.hider_alive[j]:
            ax.scatter(p[1], p[0], s=size, color=HIDER_COL, edgecolors="white", linewidths=1.2, zorder=3)
        else:
            ax.scatter(p[1], p[0], s=size, color="#777777", marker="X", zorder=3)

    # title: time, phase, whether hiders are spotted
    if env.t < cfg.HIDING_PHASE:
        phase = f"HIDING PHASE ({cfg.HIDING_PHASE - env.t} left)"
    else:
        phase = "HUNTING"
    status = "SPOTTED" if env.hiders_seen.any() else "hidden"
    alive = int(env.hider_alive.sum())
    title = f"step {env.t}/{cfg.MAX_STEPS}   {phase}   hiders alive {alive}/{env.n_hiders}   {status}"
    if env.done:
        title += f"   -> {env.winner.upper()} WIN"
    ax.set_title(title, color="#d9d4c7", fontsize=8)
    ax.axis("off")
    plt.tight_layout()

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def save_gif(frames, path, fps=6):
    images = [Image.fromarray(f) for f in frames]
    # hold the last frame a bit longer so you can read the result
    durations = [int(1000 / fps)] * (len(images) - 1) + [2000]
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0)
