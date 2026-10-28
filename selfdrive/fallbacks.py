"""Stand-ins that keep the sim running while an exercise is still a stub.

Each one is deliberately a *different* technique from the exercise it fills in
for (BFS instead of A*, pure pursuit instead of PID, ground truth instead of a
filter), so it's something to compare against, not an answer key.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

Cell = tuple[int, int]
Point = tuple[float, float]


def bfs_plan(grid: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    """Breadth-first search on a 4-connected grid. Optimal, but explores blindly."""
    rows, cols = grid.shape
    if grid[start] or grid[goal]:
        return None
    came_from: dict[Cell, Cell | None] = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = came_from[cell]
            return path[::-1]
        r, c = cell
        for nxt in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and not grid[nxt] and nxt not in came_from:
                came_from[nxt] = cell
                queue.append(nxt)
    return None


def path_distance(path: list[Point], x: float, y: float) -> float:
    """Unsigned distance from (x, y) to the nearest point on the polyline.

    Used for scoring every run the same way, regardless of whose code drove.
    """
    pts = np.asarray(path, dtype=float)
    if len(pts) == 1:
        return float(np.hypot(x - pts[0, 0], y - pts[0, 1]))
    a, b = pts[:-1], pts[1:]
    ab = b - a
    ap = np.array([x, y]) - a
    denom = np.maximum((ab**2).sum(axis=1), 1e-12)
    u = np.clip((ap * ab).sum(axis=1) / denom, 0.0, 1.0)
    closest = a + ab * u[:, None]
    return float(np.min(np.hypot(closest[:, 0] - x, closest[:, 1] - y)))


class PurePursuit:
    """Steer toward a point a fixed distance ahead on the path."""

    def __init__(self, path: list[Point], wheelbase: float, lookahead: float = 1.2):
        self.path = np.asarray(path, dtype=float)
        self.wheelbase = wheelbase
        self.lookahead = lookahead
        self.index = 0

    def steer(self, pose: tuple[float, float, float]) -> float:
        x, y, theta = pose
        pts = self.path
        # Only ever move forward along the path.
        window = pts[self.index : self.index + 20]
        self.index += int(np.argmin(np.hypot(window[:, 0] - x, window[:, 1] - y)))
        target = pts[-1]
        for p in pts[self.index :]:
            if math.hypot(p[0] - x, p[1] - y) >= self.lookahead:
                target = p
                break
        alpha = math.atan2(target[1] - y, target[0] - x) - theta
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        dist = max(math.hypot(target[0] - x, target[1] - y), 1e-6)
        return math.atan2(2.0 * self.wheelbase * math.sin(alpha), dist)


class TruthLocalizer:
    """'Localization' by asking the simulator where the car is. Cheating, but useful."""

    def __init__(self, vehicle):
        self.vehicle = vehicle

    def predict(self, steer: float, distance: float) -> None:
        pass

    def update(self, measurements) -> None:
        pass

    def estimate(self) -> tuple[float, float, float]:
        return self.vehicle.true_pose
