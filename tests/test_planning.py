import time

import numpy as np
import pytest

from selfdrive.algos.planning import astar
from selfdrive.fallbacks import bfs_plan


def assert_valid(grid, path, start, goal):
    assert path is not None, "expected a path"
    assert tuple(path[0]) == start and tuple(path[-1]) == goal
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, f"illegal move {(r1, c1)} -> {(r2, c2)}"
    for r, c in path:
        assert not grid[r, c], f"path goes through blocked cell {(r, c)}"


def test_open_grid_is_shortest():
    grid = np.zeros((5, 7), dtype=bool)
    path = astar(grid, (0, 0), (4, 6))
    assert_valid(grid, path, (0, 0), (4, 6))
    assert len(path) == 4 + 6 + 1


def test_start_equals_goal():
    grid = np.zeros((3, 3), dtype=bool)
    assert [tuple(p) for p in astar(grid, (1, 1), (1, 1))] == [(1, 1)]


def test_detours_around_wall():
    grid = np.zeros((5, 7), dtype=bool)
    grid[0:4, 3] = True  # wall with a gap at the top row only
    path = astar(grid, (0, 0), (0, 6))
    assert_valid(grid, path, (0, 0), (0, 6))
    assert len(path) == len(bfs_plan(grid, (0, 0), (0, 6)))


def test_unreachable_goal_returns_none():
    grid = np.zeros((5, 5), dtype=bool)
    grid[:, 2] = True
    assert astar(grid, (0, 0), (0, 4)) is None


def test_blocked_start_or_goal_returns_none():
    grid = np.zeros((3, 3), dtype=bool)
    grid[2, 2] = True
    assert astar(grid, (0, 0), (2, 2)) is None
    assert astar(grid, (2, 2), (0, 0)) is None


@pytest.mark.parametrize("seed", range(20))
def test_optimal_on_random_maps(seed):
    rng = np.random.default_rng(seed)
    grid = rng.random((25, 25)) < 0.28
    grid[0, 0] = grid[24, 24] = False
    expected = bfs_plan(grid, (0, 0), (24, 24))
    got = astar(grid, (0, 0), (24, 24))
    if expected is None:
        assert got is None
    else:
        assert_valid(grid, got, (0, 0), (24, 24))
        assert len(got) == len(expected), "path is valid but not shortest"


def test_does_not_modify_grid():
    grid = np.zeros((6, 6), dtype=bool)
    grid[2, 1:5] = True
    before = grid.copy()
    astar(grid, (0, 0), (5, 5))
    assert np.array_equal(grid, before)


def test_fast_on_large_grid():
    grid = np.zeros((200, 200), dtype=bool)
    for c in range(20, 180, 40):
        grid[0:180, c] = True
        grid[20:200, c + 20] = True
    t0 = time.perf_counter()
    path = astar(grid, (0, 0), (199, 199))
    assert time.perf_counter() - t0 < 1.5
    assert_valid(grid, path, (0, 0), (199, 199))
