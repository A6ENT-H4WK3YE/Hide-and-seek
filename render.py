# Draw the game as images and save GIFs: walls, vision cones, agents, where they look and who is making noise.
#
#   frame = draw_frame(env)            -> RGB numpy array of the current state
#   save_gif(frames, "runs/demo.gif")  -> save a list of frames as an animated GIF

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
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
    cell = 0.45 if env.W <= 15 else 0.22
    fig, ax = plt.subplots(figsize=(env.W * cell + 0.6, env.H * cell + 1.0), dpi=100)
    fig.patch.set_facecolor(BG)
    # map cell (row r, column c) covers x from c to c+1 and y from r to r+1, the same units the game uses
    ax.imshow(img, interpolation="nearest", extent=(0, env.W, env.H, 0))

    # thin grid lines on small maps so cells are easy to count
    if env.W <= 15:
        for k in range(env.W + 1):
            ax.axvline(k, color=BG, linewidth=0.4)
        for k in range(env.H + 1):
            ax.axhline(k, color=BG, linewidth=0.4)

    # vision cones: the agent's position plus the end of every ray (where it hits a wall)
    ns = env.n_seekers
    if show_vision:
        for j in range(env.n_hiders):
            if env.hider_alive[j]:
                ax.add_patch(Polygon(env.cone_polygon(ns + j), closed=True, facecolor=HIDER_VIEW, alpha=0.22, linewidth=0))
        for i in range(ns):
            ax.add_patch(Polygon(env.cone_polygon(i), closed=True, facecolor=SEEKER_VIEW, alpha=0.28, linewidth=0))

    # dashed lines: which seeker can see which hider
    for i in range(ns):
        for j in range(env.n_hiders):
            if env.seeker_sees[i, j]:
                (x0, y0), (x1, y1) = env.seekers[i], env.hiders[j]
                ax.plot([x0, x1], [y0, y1], "--", color=SEEKER_COL, linewidth=1.2)

    # agents: a circle the size of their body, and a short line showing where they look.
    # A dashed ring means the agent ran this step (made noise enemies nearby can hear).
    r = cfg.AGENT_RADIUS
    for team, positions, headings, noisy, colour in [
            ("seekers", env.seekers, env.seeker_heading, env.seeker_noisy, SEEKER_COL),
            ("hiders", env.hiders, env.hider_heading, env.hider_noisy, HIDER_COL)]:
        for k, (p, h) in enumerate(zip(positions, headings)):
            if team == "hiders" and not env.hider_alive[k]:
                ax.scatter(p[0], p[1], s=120 if env.W <= 15 else 40, color="#777777", marker="X", zorder=3)
                continue
            if noisy[k]:
                ax.add_patch(Circle((p[0], p[1]), 2.5 * r, fill=False, edgecolor=colour, linestyle="--", linewidth=1.0, zorder=3))
            ax.add_patch(Circle((p[0], p[1]), r, facecolor=colour, edgecolor="white", linewidth=1.0, zorder=3))
            ax.plot([p[0], p[0] + 2.2 * r * np.cos(h)], [p[1], p[1] + 2.2 * r * np.sin(h)],
                    color="white", linewidth=1.4, zorder=4)

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
    ax.set_xlim(0, env.W)
    ax.set_ylim(env.H, 0)
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
