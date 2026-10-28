"""Exercise 1: A* search on an occupancy grid.

Course tie-in: the search lesson (A*, heuristics, optimality).
Until this is implemented the sim plans with breadth-first search instead.
"""
from __future__ import annotations

import numpy as np

Cell = tuple[int, int]


def astar(grid: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    """Find a shortest path from `start` to `goal`.

    Args:
        grid: 2-D bool array, grid[row, col] is True where the cell is blocked.
        start, goal: (row, col) cells.

    Returns:
        A list of (row, col) tuples from start to goal inclusive, where each
        step moves to one of the 4 neighbors (up/down/left/right) at cost 1,
        or None if the goal can't be reached (or start/goal is blocked).
        If start == goal, return [start].

    Requirements:
        * The path must be optimal (same length as a BFS path).
        * Use an admissible heuristic. Which one is admissible for 4-connected
          unit-cost moves, and why would Euclidean distance also be admissible
          but worse?
        * Don't modify `grid`.
        * Should handle a 200x200 grid in well under a second; `heapq` is your
          friend for the open list.

    Stretch: allow diagonal moves at cost sqrt(2) behind a flag, and see what
    it does to the smoother's job.
    """
    raise NotImplementedError
