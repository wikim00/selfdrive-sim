import math

import numpy as np
import pytest

from selfdrive.algos.localization import ParticleFilter
from selfdrive.car import SimVehicle
from selfdrive.config import NoiseConfig
from selfdrive.models import Measurement, bicycle_step, expected_measurement
from selfdrive.world import World

LANDMARKS = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


def make_pf(n=1000, steering=0.02, distance=0.05, range_=0.15, bearing=0.04, seed=0, **kw):
    return ParticleFilter(LANDMARKS, n, (5.0, 5.0, 0.0), (0.3, 0.3, 0.1),
                          steering, distance, range_, bearing,
                          rng=np.random.default_rng(seed), **kw)


def exact_measurements(pose, landmarks=LANDMARKS):
    r, b = expected_measurement(*pose, landmarks[:, 0], landmarks[:, 1])
    return [Measurement(i, float(r[i]), float(b[i])) for i in range(len(landmarks))]


# ------------------------------------------------------------------ predict


def test_predict_without_noise_follows_the_model():
    pf = make_pf(n=50, steering=0.0, distance=0.0)
    before = pf.particles.copy()
    pf.predict(0.2, 1.0)
    ex, ey, et = bicycle_step(before[:, 0], before[:, 1], before[:, 2], 0.2, 1.0)
    assert np.allclose(pf.particles[:, 0], ex)
    assert np.allclose(pf.particles[:, 1], ey)
    assert np.allclose(np.cos(pf.particles[:, 2]), np.cos(et))
    assert np.allclose(np.sin(pf.particles[:, 2]), np.sin(et))


def test_predict_with_noise_spreads_identical_particles():
    pf = make_pf(n=500)
    pf.particles[:] = [5.0, 5.0, 0.0]
    pf.predict(0.1, 2.0)
    assert pf.particles[:, 0].std() > 0.01
    assert pf.particles[:, 2].std() > 0.005
    assert pf.particles[:, 0].mean() == pytest.approx(bicycle_step(5, 5, 0, 0.1, 2.0)[0], abs=0.05)


def test_predict_keeps_heading_wrapped():
    pf = make_pf(n=200)
    pf.particles[:, 2] = math.pi - 0.01
    pf.predict(0.5, 1.0)
    assert np.all(pf.particles[:, 2] >= -math.pi) and np.all(pf.particles[:, 2] < math.pi)


# ------------------------------------------------------------------- update


def test_update_prefers_the_particle_that_explains_the_data():
    pf = make_pf(n=2)
    truth = (5.0, 5.0, 0.3)
    pf.particles = np.array([truth, (6.0, 5.0, 0.3)])
    pf.weights = np.array([0.5, 0.5])
    pf.update(exact_measurements(truth))
    assert pf.weights.sum() == pytest.approx(1.0)
    assert pf.weights[0] > 10 * pf.weights[1]


def test_update_wraps_bearing_error():
    pf = make_pf(n=3)
    truth = (5.0, 5.0, math.pi - 0.01)
    pf.particles = np.array([truth, (5.0, 5.0, -math.pi + 0.01), (5.0, 5.0, 0.0)])
    pf.weights = np.full(3, 1 / 3)
    pf.update(exact_measurements(truth))
    # Particle 1 faces (almost) the same way as the truth; particle 2 faces the opposite way.
    assert pf.weights[1] > 0.5 * pf.weights[0]
    assert pf.weights[1] > 10 * pf.weights[2]


def test_update_with_no_measurements_keeps_weights():
    pf = make_pf(n=4)
    pf.weights = np.array([0.1, 0.2, 0.3, 0.4])
    pf.update([])
    assert np.allclose(pf.weights, [0.1, 0.2, 0.3, 0.4])


def test_update_does_not_underflow_when_everyone_is_wrong():
    landmarks = np.random.default_rng(1).uniform(0, 20, size=(40, 2))
    pf = ParticleFilter(landmarks, 100, (10.0, 10.0, 0.0), (0.1, 0.1, 0.01),
                        0.0, 0.0, 0.05, 0.01, rng=np.random.default_rng(0))
    truth = (13.0, 13.0, 0.0)  # every particle is ~4 m off with very confident sensors
    pf.update(exact_measurements(truth, landmarks))
    assert np.all(np.isfinite(pf.weights))
    assert pf.weights.sum() == pytest.approx(1.0)
    best = pf.particles[np.argmax(pf.weights)]
    worst = pf.particles[np.argmin(pf.weights)]
    assert math.dist(best[:2], truth[:2]) <= math.dist(worst[:2], truth[:2])


# ----------------------------------------------------------------- resample


def test_resample_keeps_count_and_resets_weights():
    pf = make_pf(n=300)
    pf.weights = np.random.default_rng(3).random(300)
    pf.weights /= pf.weights.sum()
    pf.resample()
    assert pf.particles.shape == (300, 3)
    assert np.allclose(pf.weights, 1 / 300)


def test_resample_favors_heavy_particles():
    pf = make_pf(n=200)
    pf.particles[0] = [42.0, 42.0, 1.0]
    pf.weights = np.full(200, 0.03 / 199)
    pf.weights[0] = 0.97
    pf.resample()
    hits = np.all(np.isclose(pf.particles, [42.0, 42.0, 1.0]), axis=1).sum()
    assert hits > 0.9 * 200


def test_resample_returns_independent_copies():
    pf = make_pf(n=10)
    pf.weights = np.zeros(10)
    pf.weights[0] = 1.0
    pf.resample()
    pf.particles[0, 0] = -99.0
    assert pf.particles[1, 0] != -99.0


# ----------------------------------------------------------------- estimate


def test_estimate_is_weighted_mean():
    pf = make_pf(n=2)
    pf.particles = np.array([[0.0, 0.0, 0.0], [4.0, 8.0, 0.0]])
    pf.weights = np.array([0.75, 0.25])
    x, y, _ = pf.estimate()
    assert (x, y) == pytest.approx((1.0, 2.0))


def test_estimate_heading_is_circular():
    pf = make_pf(n=2)
    pf.particles = np.array([[0.0, 0.0, math.pi - 0.1], [0.0, 0.0, -math.pi + 0.1]])
    pf.weights = np.array([0.5, 0.5])
    _, _, theta = pf.estimate()
    assert abs(theta) > math.pi - 0.05


# -------------------------------------------------------------- integration


ARENA = "\n".join(
    ["#" * 20]
    + ["#" + "." * 18 + "#"] * 3
    + ["#...L.........L....#"]
    + ["#" + "." * 18 + "#"] * 5
    + ["#........S.........#"]
    + ["#" + "." * 18 + "#"] * 4
    + ["#...L.........L..G.#"]
    + ["#" + "." * 18 + "#"] * 3
    + ["#" * 20]
)


def test_tracks_a_simulated_car():
    world = World.from_ascii(ARENA)
    noise = NoiseConfig()
    rng = np.random.default_rng(7)
    start = (10.0, 9.5, 0.0)
    car = SimVehicle(world, start, noise, rng, sensor_range=20.0)
    pf = ParticleFilter(world.landmarks, 500, start, (0.3, 0.3, 0.1),
                        noise.steering, noise.distance, noise.range, noise.bearing, rng=rng)
    errors = []
    for _ in range(200):  # drive in a circle
        car.drive(0.2, 0.2)
        pf.predict(0.2, 0.2)
        pf.update(car.sense())
        pf.resample()
        x, y, _ = pf.estimate()
        errors.append(math.hypot(x - car.x, y - car.y))
    assert np.mean(errors) < 0.3
