# Draw the game as images and save GIFs: walls, sites, vision cones, players and where they look, who is being
# watched (and how close to being taken out), the spike,
# and sound: a running (or planting / defusing) player gets a dashed ring plus a faint dotted circle showing how far
# the noise carries, and a yellow dotted line joins every player to the enemy noise it hears.
#
#   frame = draw_frame(env)            -> RGB numpy array of the current state
#   save_gif(frames, "runs/demo.gif")  -> save a list of frames as an animated GIF (3 frames per second)
#
# Colours: attackers red, defenders blue. An orange ring = an enemy has this player in its sights; the thicker the
# ring, the closer it is to being taken out. Grey X = out. The spike is a yellow star.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from PIL import Image
import config as cfg
from env import TEAMS, OTHER

WALL = np.array([43, 47, 58]) / 255
FLOOR = np.array([217, 212, 199]) / 255
SITE = np.array([230, 201, 143]) / 255
ATTACKER_SPAWN = np.array([236, 214, 208]) / 255
DEFENDER_SPAWN = np.array([210, 220, 228]) / 255
COLOUR = {"attackers": "#c0392b", "defenders": "#2563b0"}
VIEW = {"attackers": np.array([220, 70, 60]) / 255, "defenders": np.array([60, 120, 220]) / 255}
SOUND_COL = "#e0b000"     # noise: hearing range circles and "who hears whom" lines
WATCHED_COL = "#ff8c00"   # being watched (kept in an enemy's sights)
BG = "#14161c"


def base_image(env):
    # the map itself: walls, floor, spawn areas, sites
    img = np.zeros((env.H, env.W, 3))
    img[:] = FLOOR
    img[env.grid == "#"] = WALL
    img[env.grid == "s"] = ATTACKER_SPAWN
    img[env.grid == "h"] = DEFENDER_SPAWN
    img[env.sites] = SITE
    return img


def draw_frame(env, show_vision=True):
    img = base_image(env)
    cell = 0.45 if env.W <= 15 else 0.22
    fig, ax = plt.subplots(figsize=(env.W * cell + 0.6, env.H * cell + 1.0), dpi=100)
    fig.patch.set_facecolor(BG)
    # map cell (row r, column c) covers x from c to c+1 and y from r to r+1, the same units the game uses
    ax.imshow(img, interpolation="nearest", extent=(0, env.W, env.H, 0))
    for site in "ABC":
        cells = np.argwhere(env.grid == site)
        if len(cells):
            ax.text(cells[:, 1].mean() + 0.5, cells[:, 0].mean() + 0.5, site, color="#8a6d2b", fontsize=14,
                    ha="center", va="center", fontweight="bold", alpha=0.6)

    # vision cones: the player's position plus the end of every ray (where it hits a wall)
    if show_vision:
        for team in TEAMS:
            for i in range(env.n[team]):
                if env.alive[team][i]:
                    ax.add_patch(Polygon(env.cone_polygon(env.offset[team] + i), closed=True, facecolor=VIEW[team],
                                         alpha=0.22, linewidth=0))

    # sound: how far each noisy player can be heard, and who hears whom (yellow dotted lines)
    for team in TEAMS:
        for i, p in enumerate(env.pos[team]):
            if env.noisy[team][i] and env.alive[team][i]:
                ax.add_patch(Circle((p[0], p[1]), env.hearing_range, fill=False, edgecolor=SOUND_COL, linestyle=":",
                                    linewidth=0.8, alpha=0.6, zorder=2))
        for i in range(env.n[team]):
            for j in range(env.n[OTHER[team]]):
                if env.hears[team][i, j]:
                    a, b = env.pos[team][i], env.pos[OTHER[team]][j]
                    ax.plot([a[0], b[0]], [a[1], b[1]], ":", color=SOUND_COL, linewidth=1.4, zorder=2)

    # dashed lines: who sees which enemy (in the viewer's colour)
    for team in TEAMS:
        for i in range(env.n[team]):
            for j in range(env.n[OTHER[team]]):
                if env.sees[team][i, j]:
                    a, b = env.pos[team][i], env.pos[OTHER[team]][j]
                    ax.plot([a[0], b[0]], [a[1], b[1]], "--", color=COLOUR[team], linewidth=1.0, alpha=0.8)

    # the spike
    if env.spike is not None:
        ax.scatter(env.spike[0], env.spike[1], s=260 if env.W <= 15 else 120, marker="*", color="#ffd400",
                   edgecolors="black", linewidths=0.8, zorder=5)

    # players: a circle the size of their body and a short line showing where they look; dashed ring = noisy,
    # orange ring = being watched (thicker = closer to being taken out), grey X = out
    r = cfg.AGENT_RADIUS
    for team in TEAMS:
        for i, (p, h) in enumerate(zip(env.pos[team], env.heading[team])):
            if not env.alive[team][i]:
                ax.scatter(p[0], p[1], s=120 if env.W <= 15 else 40, color="#777777", marker="X", zorder=3)
                continue
            if env.noisy[team][i]:
                ax.add_patch(Circle((p[0], p[1]), 2.5 * r, fill=False, edgecolor=COLOUR[team], linestyle="--",
                                    linewidth=1.0, zorder=3))
            danger = env.watch_progress(OTHER[team], target=i)
            if danger > 0:
                ax.add_patch(Circle((p[0], p[1]), 1.6 * r, fill=False, edgecolor=WATCHED_COL, linewidth=0.5 + 3.5 * danger,
                                    zorder=4))
            ax.add_patch(Circle((p[0], p[1]), r, facecolor=COLOUR[team], edgecolor="white", linewidth=1.0, zorder=4))
            ax.plot([p[0], p[0] + 2.2 * r * np.cos(h)], [p[1], p[1] + 2.2 * r * np.sin(h)],
                    color="white", linewidth=1.4, zorder=5)

    # title: time, phase, players left, plant / spike / defuse
    phase = f"SETUP ({cfg.SETUP_STEPS - env.t} left)" if env.t < cfg.SETUP_STEPS else "ROUND"
    left = f"attackers {int(env.alive['attackers'].sum())}/{env.n['attackers']}  defenders {int(env.alive['defenders'].sum())}/{env.n['defenders']}"
    title = f"step {env.t}   {phase}   {left}"
    if env.spike is not None:
        title += f"   SPIKE DOWN ({env.spike_timer} left)"
        if env.defuse.max() > 0:
            title += f"   defusing {env.defuse.max()}/{cfg.DEFUSE_STEPS}"
    elif env.plant["attackers"].max() > 0:
        title += f"   planting {env.plant['attackers'].max()}/{cfg.PLANT_STEPS}"
    if env.done:
        title += f"   -> {env.winner.upper()} WIN" if env.winner != "draw" else "   -> DRAW"
    ax.set_title(title, color="#d9d4c7", fontsize=8)
    ax.set_xlim(0, env.W)
    ax.set_ylim(env.H, 0)
    ax.axis("off")
    plt.tight_layout()

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def save_gif(frames, path, fps=3):     # 3 frames per second: half the old speed, easier to follow
    images = [Image.fromarray(f) for f in frames]
    # hold the last frame a bit longer so you can read the result
    durations = [int(1000 / fps)] * (len(images) - 1) + [2000]
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0)
