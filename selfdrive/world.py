"""The world: an occupancy grid of walls, landmark posts, a start and a goal.

Maps are plain text files, drawn the way they appear on screen:

    #  wall            S  start
    .  open floor      G  goal
    L  landmark post (open floor with a landmark at the cell center)

Grid cells are indexed (row, col) with row 0 at the *bottom*, so cell (r, c)
covers x in [c, c+1) and y in [r, r+1), and its center is (c + 0.5, r + 0.5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

Cell = tuple[int, int]


class World:
    def __init__(self, grid: np.ndarray, landmarks: np.ndarray, start: Cell, goal: Cell):
        self.grid = np.asarray(grid, dtype=bool)  # True = wall
        self.landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)
        self.start = start
        self.goal = goal

    @classmethod
    def from_ascii(cls, text: str) -> "World":
        lines = [ln.rstrip() for ln in text.strip("\n").splitlines() if ln.strip()]
        width = len(lines[0])
        if any(len(ln) != width for ln in lines):
            raise ValueError("every map row must be the same width")
        lines.reverse()  # first text line is the top of the map

        grid = np.zeros((len(lines), width), dtype=bool)
        landmarks, start, goal = [], None, None
        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                if ch == "#":
                    grid[r, c] = True
                elif ch == "L":
                    landmarks.append((c + 0.5, r + 0.5))
                elif ch == "S":
                    start = (r, c)
                elif ch == "G":
                    goal = (r, c)
                elif ch != ".":
                    raise ValueError(f"unknown map character {ch!r}")
        if start is None or goal is None:
            raise ValueError("map needs one S and one G")
        return cls(grid, np.array(landmarks), start, goal)

    @classmethod
    def from_file(cls, path: str | Path) -> "World":
        return cls.from_ascii(Path(path).read_text())

    @property
    def rows(self) -> int:
        return self.grid.shape[0]

    @property
    def cols(self) -> int:
        return self.grid.shape[1]

    @staticmethod
    def cell_center(cell: Cell) -> tuple[float, float]:
        r, c = cell
        return (c + 0.5, r + 0.5)

    @staticmethod
    def cell_of(x: float, y: float) -> Cell:
        return (int(np.floor(y)), int(np.floor(x)))

    def is_free(self, x: float, y: float) -> bool:
        r, c = self.cell_of(x, y)
        return 0 <= r < self.rows and 0 <= c < self.cols and not self.grid[r, c]


def inflate(grid: np.ndarray, margin: int) -> np.ndarray:
    """Block every cell within `margin` cells (including diagonals) of a wall.

    Planning on the inflated grid keeps the route away from walls, so the car
    has room to wobble while tracking it. The area off the edge of the map
    counts as wall.
    """
    grid = np.asarray(grid, dtype=bool)
    if margin <= 0:
        return grid.copy()
    padded = np.pad(grid, margin, constant_values=True)
    out = np.zeros_like(grid)
    rows, cols = grid.shape
    for dr in range(-margin, margin + 1):
        for dc in range(-margin, margin + 1):
            out |= padded[margin + dr : margin + dr + rows, margin + dc : margin + dc + cols]
    return out
