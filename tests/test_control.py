import math

import pytest

from selfdrive.algos.control import PIDController, cross_track_error

STRAIGHT = [(0.0, 0.0), (10.0, 0.0)]
L_TURN = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]


# ------------------------------------------------------- cross-track error


def test_on_path_is_zero():
    cte, idx = cross_track_error(STRAIGHT, 3.0, 0.0, 0)
    assert cte == pytest.approx(0.0) and idx == 0


def test_left_is_positive_right_is_negative():
    assert cross_track_error(STRAIGHT, 3.0, 0.5, 0)[0] == pytest.approx(0.5)
    assert cross_track_error(STRAIGHT, 3.0, -0.5, 0)[0] == pytest.approx(-0.5)


def test_sign_follows_direction_of_travel():
    backwards = [(10.0, 0.0), (0.0, 0.0)]
    # Driving toward -x, a car at y = +0.5 is on the RIGHT.
    assert cross_track_error(backwards, 3.0, 0.5, 0)[0] == pytest.approx(-0.5)


def test_diagonal_segment():
    cte, _ = cross_track_error([(0.0, 0.0), (4.0, 4.0)], 0.0, 1.0, 0)
    assert cte == pytest.approx(math.sqrt(2) / 2)


def test_advances_past_end_of_segment():
    cte, idx = cross_track_error(L_TURN, 5.5, 2.0, 0)
    assert idx == 1
    assert cte == pytest.approx(-0.5)  # right of the upward segment


def test_skips_several_segments_if_needed():
    path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
    _, idx = cross_track_error(path, 3.5, 0.2, 0)
    assert idx == 3


def test_never_goes_backward():
    cte, idx = cross_track_error(L_TURN, 1.0, 0.3, 1)
    assert idx == 1
    assert cte == pytest.approx(4.0)  # left of the upward segment


def test_does_not_run_off_the_last_segment():
    _, idx = cross_track_error(L_TURN, 5.2, 9.0, 0)
    assert idx == len(L_TURN) - 2


# --------------------------------------------------------------------- PID


def test_proportional_only():
    assert PIDController(2.0, 0.0, 0.0).step(0.5) == pytest.approx(-1.0)


def test_no_derivative_kick_on_first_step():
    assert PIDController(0.0, 0.0, 5.0).step(1.0) == pytest.approx(0.0)


def test_derivative():
    pid = PIDController(0.0, 0.0, 2.0)
    pid.step(1.0)
    assert pid.step(0.5) == pytest.approx(1.0)  # -(2 * (0.5 - 1.0))


def test_integral_accumulates():
    pid = PIDController(0.0, 1.0, 0.0)
    assert pid.step(1.0) == pytest.approx(-1.0)
    assert pid.step(1.0) == pytest.approx(-2.0)


def test_dt_scales_integral_and_derivative():
    pid = PIDController(0.0, 1.0, 1.0)
    assert pid.step(1.0, dt=0.5) == pytest.approx(-0.5)
    # integral = 0.5 + 1.0 = 1.5, derivative = (2 - 1) / 0.5 = 2
    assert pid.step(2.0, dt=0.5) == pytest.approx(-3.5)


def test_reset_clears_state():
    pid = PIDController(1.0, 1.0, 1.0)
    pid.step(3.0)
    pid.step(1.0)
    pid.reset()
    assert pid.step(1.0) == pytest.approx(PIDController(1.0, 1.0, 1.0).step(1.0))


def test_steers_back_toward_path():
    pid = PIDController(0.5, 0.0, 2.0)
    assert pid.step(0.4) < 0  # left of path -> steer right
