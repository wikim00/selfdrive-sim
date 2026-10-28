import pytest

from selfdrive.algos.tuning import twiddle


def bowl(p):
    return (p[0] - 1.0) ** 2 + 2 * (p[1] + 2.0) ** 2 + 3.0


def test_finds_minimum_of_a_bowl():
    best, cost = twiddle(bowl, [0.0, 0.0], [1.0, 1.0], tolerance=1e-5)
    assert best[0] == pytest.approx(1.0, abs=1e-2)
    assert best[1] == pytest.approx(-2.0, abs=1e-2)
    assert cost == pytest.approx(3.0, abs=1e-3)


def test_reported_cost_matches_params():
    best, cost = twiddle(bowl, [5.0, 5.0], [1.0, 1.0], tolerance=1e-4)
    assert cost == pytest.approx(bowl(best))


def test_inputs_not_modified():
    params, deltas = [0.0, 0.0], [1.0, 1.0]
    twiddle(bowl, params, deltas, tolerance=1e-3)
    assert params == [0.0, 0.0] and deltas == [1.0, 1.0]


def test_already_optimal_stays_put():
    best, _ = twiddle(bowl, [1.0, -2.0], [0.5, 0.5], tolerance=1e-4)
    assert best == pytest.approx([1.0, -2.0], abs=1e-6)


def test_respects_max_iterations():
    calls = []

    def counting(p):
        calls.append(1)
        return bowl(p)

    twiddle(counting, [100.0, 100.0], [1.0, 1.0], tolerance=0.0, max_iterations=3)
    assert len(calls) <= 1 + 3 * 2 * 2  # initial + sweeps * params * (up, down)
