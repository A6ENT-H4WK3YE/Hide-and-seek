# Geometry for the continuous game: moving round agents among square walls, and casting rays.
#
# Coordinates: x to the right, y down (like the map file, row 0 at the top).
# Map cell (row r, column c) covers x from c to c+1 and y from r to r+1, so its centre is (c + 0.5, r + 0.5).
# Every '#' cell is a solid 1 x 1 square. Outside the map also counts as wall.
# Angles are in radians, 0 = facing right (east). Because y points down, a positive angle turns clockwise on screen.
#
#   move_circle(walls, x, y, dx, dy, radius, substeps)   -> new (x, y), sliding along walls
#   cast_rays(walls, ox, oy, angles)                      -> distance to the first wall along each ray
#   line_clear(walls, x0, y0, x1, y1)                     -> True where nothing solid is in between

import numpy as np


def is_wall(walls, r, c):
    H, W = walls.shape
    return r < 0 or r >= H or c < 0 or c >= W or bool(walls[r, c])


def push_out(walls, x, y, radius):
    # if the circle overlaps any wall square, push it out along the shortest way.
    # Checking the squares one by one also handles inside corners (two walls at right angles).
    # The closest square goes first: along a flat wall made of several squares, the seam between
    # two squares would otherwise act like a corner and nudge the agent sideways.
    squares = []
    for r in range(int(np.floor(y - radius)), int(np.floor(y + radius)) + 1):
        for c in range(int(np.floor(x - radius)), int(np.floor(x + radius)) + 1):
            if is_wall(walls, r, c):
                cx, cy = min(max(x, c), c + 1), min(max(y, r), r + 1)
                squares.append(((x - cx) ** 2 + (y - cy) ** 2, r, c))
    for _, r, c in sorted(squares):
        # closest point of the square to the circle's centre (worked out again, the circle may have moved)
        cx, cy = min(max(x, c), c + 1), min(max(y, r), r + 1)
        ddx, ddy = x - cx, y - cy
        d2 = ddx * ddx + ddy * ddy
        if 1e-12 < d2 < radius * radius:
            d = np.sqrt(d2)
            x, y = cx + ddx / d * radius, cy + ddy / d * radius
    return x, y


def overlaps_wall(walls, x, y, radius, tol=1e-6):
    for r in range(int(np.floor(y - radius)), int(np.floor(y + radius)) + 1):
        for c in range(int(np.floor(x - radius)), int(np.floor(x + radius)) + 1):
            if is_wall(walls, r, c):
                cx, cy = min(max(x, c), c + 1), min(max(y, r), r + 1)
                if (x - cx) ** 2 + (y - cy) ** 2 < (radius - tol) ** 2:
                    return True
    return False


def move_circle(walls, x, y, dx, dy, radius, substeps):
    # move in a few small steps, pushing out of walls after each one. Steps are shorter than
    # the radius, so a circle can never jump through a wall or slip through a corner.
    # Pushing out keeps the part of the move along the wall, so agents slide instead of sticking.
    for _ in range(substeps):
        x, y = push_out(walls, x + dx / substeps, y + dy / substeps, radius)
    return x, y


def cast_rays(walls, ox, oy, angles, eps=1e-9):
    # Distance from each origin (ox, oy) along each angle to the first wall it hits.
    # ox, oy, angles: arrays of the same length (one entry per ray), all rays traced at once.
    # Walks the ray cell by cell across the grid (the "DDA" method used by old 3D shooters),
    # so it is exact for square walls and never looks past the first wall.
    # A ray passing exactly through a corner where two walls meet counts as blocked.
    H, W = walls.shape
    ox, oy, angles = np.broadcast_arrays(np.asarray(ox, float), np.asarray(oy, float), np.asarray(angles, float))
    ox, oy, angles = ox.ravel(), oy.ravel(), angles.ravel()
    n = len(angles)
    dx, dy = np.cos(angles), np.sin(angles)
    cx, cy = np.floor(ox).astype(int), np.floor(oy).astype(int)
    sx, sy = np.where(dx > 0, 1, -1), np.where(dy > 0, 1, -1)
    with np.errstate(divide="ignore"):
        tdx = np.where(np.abs(dx) > eps, 1.0 / np.abs(dx), np.inf)   # ray length to cross one cell in x
        tdy = np.where(np.abs(dy) > eps, 1.0 / np.abs(dy), np.inf)
    # ray length to reach the next vertical / horizontal grid line
    tx = np.where(dx > 0, (cx + 1 - ox), (ox - cx)) * tdx
    ty = np.where(dy > 0, (cy + 1 - oy), (oy - cy)) * tdy
    tx = np.where(np.isfinite(tdx), tx, np.inf)
    ty = np.where(np.isfinite(tdy), ty, np.inf)

    padded = np.ones((H + 2, W + 2), dtype=bool)   # border of wall so rays never leave the map
    padded[1:-1, 1:-1] = walls
    dist = np.full(n, np.inf)
    active = np.ones(n, dtype=bool)
    for _ in range(H + W + 2):
        if not active.any():
            break
        corner = np.abs(tx - ty) < 1e-9
        step_x = (tx < ty) & ~corner
        step_y = (ty < tx) & ~corner
        t = np.minimum(tx, ty)
        # exactly through a corner: blocked if either side cell is a wall, otherwise go diagonally
        # (indices clipped only so finished rays sitting on the border never index outside the array)
        side_x = padded[np.clip(cy + 1, 0, H + 1), np.clip(cx + sx + 1, 0, W + 1)]
        side_y = padded[np.clip(cy + sy + 1, 0, H + 1), np.clip(cx + 1, 0, W + 1)]
        corner_blocked = active & corner & (side_x | side_y)
        dist[corner_blocked] = t[corner_blocked]
        active &= ~corner_blocked
        move_x = active & (step_x | corner)
        move_y = active & (step_y | corner)
        cx = np.where(move_x, cx + sx, cx)
        cy = np.where(move_y, cy + sy, cy)
        tx = np.where(move_x, tx + tdx, tx)
        ty = np.where(move_y, ty + tdy, ty)
        hit = active & padded[np.clip(cy + 1, 0, H + 1), np.clip(cx + 1, 0, W + 1)]
        dist[hit] = t[hit]
        active &= ~hit
    return dist


def line_clear(walls, x0, y0, x1, y1):
    # True where the straight line from (x0, y0) to (x1, y1) does not touch a wall (arrays allowed)
    x0, y0, x1, y1 = (np.asarray(v, float) for v in (x0, y0, x1, y1))
    length = np.hypot(x1 - x0, y1 - y0)
    wall_dist = cast_rays(walls, x0, y0, np.arctan2(y1 - y0, x1 - x0))
    return wall_dist.reshape(length.shape) >= length


def wrap_angle(a):
    # any angle -> the same direction between -pi and +pi
    return (a + np.pi) % (2 * np.pi) - np.pi
