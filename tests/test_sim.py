"""Tests for the given simulator code. These pass before you write anything."""
import math
from dataclasses import replace

import numpy as np
import pytest

from selfdrive.config import SimConfig
from selfdrive.fallbacks import bfs_plan, path_distance
from selfdrive.models import bicycle_step, expected_measurement, wrap_angle
from selfdrive.run import COMPONENTS, run
from selfdrive.world import World, inflate


def test_bicycle_straight():
    assert bicycle_step(0, 0, 0, 0, 2.0) == pytest.approx((2.0, 0.0, 0.0))


def test_bicycle_quarter_turn():
    wheelbase = 1.0
    steer = math.atan(1.0)  # turning radius = wheelbase / tan(steer) = 1
    x, y, t = bicycle_step(0, 0, 0, steer, math.pi / 2, wheelbase)
    assert (x, y, t) == pytest.approx((1.0, 1.0, math.pi / 2))


def test_bicycle_vectorized():
    xs, ys, ts = bicycle_step(np.zeros(3), np.zeros(3), np.zeros(3), np.array([0.0, 0.2, -0.2]), 1.0)
    assert xs.shape == (3,) and ys[1] > 0 > ys[2]


def test_wrap_angle():
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert wrap_angle(-3 * math.pi / 2) == pytest.approx(math.pi / 2)


def test_expected_measurement():
    r, b = expected_measurement(0, 0, math.pi / 2, 1.0, 1.0)
    assert r == pytest.approx(math.sqrt(2))
    assert b == pytest.approx(-math.pi / 4)


def test_default_map_is_solvable():
    world = World.from_file(SimConfig.map_path)
    grid = inflate(world.grid, SimConfig.margin)
    assert not grid[world.start] and not grid[world.goal]
    assert bfs_plan(grid, world.start, world.goal) is not None
    assert len(world.landmarks) >= 6


def test_inflate():
    grid = np.zeros((7, 7), dtype=bool)
    grid[3, 3] = True
    out = inflate(grid, 1)
    assert out[2:5, 2:5].all()
    assert out[1:6, 1:6].sum() == 9  # ring around the 3x3 stays open
    assert out[0].all() and out[:, 0].all()  # off-map counts as wall


def test_path_distance():
    assert path_distance([(0, 0), (4, 0), (4, 4)], 2.0, 1.0) == pytest.approx(1.0)


def test_stand_ins_alone_reach_the_goal():
    result = run(replace(SimConfig(), force_fallback=frozenset(COMPONENTS)))
    assert result.outcome == "goal"
