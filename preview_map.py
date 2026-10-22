# Draw a map file to a PNG and check that every walkable cell is reachable.
# Usage:  python preview_map.py maps/tiny.txt
#
# Map legend:
#   #        wall
#   .        floor
#   h        defender spawn cell (top)
#   s        attacker spawn cell (bottom)
#   A B C    site cells (walkable, just labelled)

import sys
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

path = sys.argv[1] if len(sys.argv) > 1 else "maps/tiny.txt"
rows = [r.rstrip("\n") for r in open(path) if r.strip()]
grid = np.array([list(r) for r in rows])
H, W = grid.shape

# ---------- connectivity check ----------
free = grid != "#"
start = tuple(np.argwhere(free)[0])
seen = {start}
queue = deque([start])
while queue:
    y, x = queue.popleft()
    for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W and free[ny, nx] and (ny, nx) not in seen:
            seen.add((ny, nx))
            queue.append((ny, nx))
print(f"{path}: {W} x {H}, walkable cells {free.sum()}, reachable {len(seen)}")
if len(seen) != free.sum():
    print("WARNING: some walkable cells cannot be reached")
print(f"defender spawn cells {(grid == 'h').sum()}, attacker spawn cells {(grid == 's').sum()}")

# ---------- draw ----------
colours = {
    "#": "#2b2f3a",
    ".": "#d9d4c7",
    "h": "#9fc3e6",
    "s": "#e6a19f",
    "A": "#e6c98f",
    "B": "#e6c98f",
    "C": "#e6c98f",
}

cell = 0.25 if W > 20 else 0.5
fig, ax = plt.subplots(figsize=(W * cell + 1, H * cell + 1), dpi=130)
fig.patch.set_facecolor("#14161c")
ax.set_facecolor("#14161c")

for y in range(H):
    for x in range(W):
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=colours.get(grid[y, x], "#d9d4c7"),
                               edgecolor="#14161c", linewidth=0.3))

# label each site and spawn zone at its centre
labels = {"A": "A", "B": "B", "C": "C", "h": "HIDERS", "s": "SEEKERS"}
for ch, text in labels.items():
    cells = np.argwhere(grid == ch)
    if len(cells):
        cy, cx = cells.mean(axis=0)
        size = 20 if ch in "ABC" else 9
        ax.text(cx + 0.5, cy + 0.5, text, ha="center", va="center",
                fontsize=size, fontweight="bold", color="#14161c")

ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(f"{path}  ({W} x {H})", color="#d9d4c7", fontsize=11, pad=8)
plt.tight_layout()
out = path.replace(".txt", "_preview.png")
plt.savefig(out, facecolor=fig.get_facecolor())
print("saved", out)
