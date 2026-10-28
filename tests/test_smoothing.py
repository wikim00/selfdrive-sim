import numpy as np

from selfdrive.algos.smoothing import smooth

L_PATH = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0),
          (4.0, 1.0), (4.0, 2.0), (4.0, 3.0), (4.0, 4.0)]


def roughness(points):
    p = np.asarray(points, dtype=float)
    return float((np.diff(p, n=2, axis=0) ** 2).sum())


def test_same_length_and_float_pairs():
    out = smooth(L_PATH)
    assert len(out) == len(L_PATH)
    for p in out:
        assert len(p) == 2
        float(p[0]), float(p[1])


def test_endpoints_fixed():
    out = smooth(L_PATH, 0.5, 0.3)
    assert np.allclose(out[0], L_PATH[0])
    assert np.allclose(out[-1], L_PATH[-1])


def test_input_not_modified():
    path = [list(p) for p in L_PATH]
    before = [list(p) for p in path]
    smooth(path)
    assert path == before


def test_zero_smoothing_is_identity():
    assert np.allclose(smooth(L_PATH, 0.5, 0.0), L_PATH)


def test_straight_line_unchanged():
    line = [(float(i), 2.0 * i) for i in range(8)]
    assert np.allclose(smooth(line, 0.5, 0.3), line, atol=1e-4)


def test_corner_is_cut_toward_inside():
    out = np.asarray(smooth(L_PATH, 0.5, 0.2))
    corner = out[4]
    # The corner at (4, 0) should move up-left, toward the inside of the turn.
    assert corner[0] < 4.0 - 0.05 and corner[1] > 0.05
    assert roughness(out) < roughness(L_PATH)


def test_more_smoothing_is_smoother_but_further_away():
    gentle = np.asarray(smooth(L_PATH, 0.5, 0.05))
    strong = np.asarray(smooth(L_PATH, 0.5, 0.4))
    assert roughness(strong) < roughness(gentle)
    orig = np.asarray(L_PATH)
    assert np.abs(strong - orig).max() > np.abs(gentle - orig).max()


def test_stays_near_original():
    out = np.asarray(smooth(L_PATH, 0.5, 0.2))
    assert np.abs(out - np.asarray(L_PATH)).max() < 1.0


def test_converged():
    # A tighter tolerance should barely change the answer.
    a = np.asarray(smooth(L_PATH, 0.5, 0.2, tolerance=1e-6))
    b = np.asarray(smooth(L_PATH, 0.5, 0.2, tolerance=1e-10))
    assert np.abs(a - b).max() < 1e-3
