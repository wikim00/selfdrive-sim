"""Exercise 5: Monte Carlo localization (particle filter).

Course tie-in: the particle filter lesson (motion update, measurement
weights, resampling). Until this is implemented the sim "localizes" by reading
the true pose straight from the simulator, which no real car can do.
"""
from __future__ import annotations

import numpy as np

from ..models import WHEELBASE, Measurement, bicycle_step, expected_measurement, wrap_angle  # noqa: F401


class ParticleFilter:
    """Tracks the car's pose (x, y, theta) with a cloud of weighted guesses.

    State you work with:
        self.particles  (N, 3) float array of [x, y, theta]
        self.weights    (N,)   float array, sums to 1

    Given helpers (in selfdrive/models.py): bicycle_step moves arrays of poses,
    expected_measurement gives noise-free (range, bearing) for arrays of poses,
    wrap_angle keeps angles in [-pi, pi).
    """

    def __init__(
        self,
        landmarks: np.ndarray,
        n_particles: int,
        initial_pose: tuple[float, float, float],
        init_std: tuple[float, float, float],
        steering_noise: float,
        distance_noise: float,
        range_noise: float,
        bearing_noise: float,
        rng: np.random.Generator | None = None,
        wheelbase: float = WHEELBASE,
    ):
        self.landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)  # (K, 2), id = row index
        self.n = n_particles
        self.steering_noise = steering_noise  # rad std
        self.distance_noise = distance_noise  # fractional std of distance
        self.range_noise = range_noise  # m std
        self.bearing_noise = bearing_noise  # rad std
        self.rng = rng if rng is not None else np.random.default_rng()
        self.wheelbase = wheelbase

        x0, y0, t0 = initial_pose
        sx, sy, st = init_std
        self.particles = np.column_stack(
            [
                self.rng.normal(x0, sx, self.n),
                self.rng.normal(y0, sy, self.n),
                wrap_angle(self.rng.normal(t0, st, self.n)),
            ]
        )
        self.weights = np.full(self.n, 1.0 / self.n)

    def predict(self, steer: float, distance: float) -> None:
        """Motion update: move every particle as the car was commanded to move.

        Each particle gets its own random steering and distance error drawn
        with the same noise model the car has (see SimVehicle.drive). With
        both noise levels at 0, every particle must move exactly as
        bicycle_step says. Vectorize: no Python loop over particles.
        """
        raise NotImplementedError

    def update(self, measurements: list[Measurement]) -> None:
        """Measurement update: reweight particles by how well they explain the sightings.

        For each measurement, compare what the particle *would* see from its
        pose (expected_measurement) with what the car actually saw, scored
        with Gaussian likelihoods on range and bearing error. Combine across
        measurements, then normalize so weights sum to 1.

        Requirements:
          * If measurements is empty, leave the weights unchanged.
          * Bearing errors must be wrapped: 3.13 vs -3.13 rad is a 0.02 rad miss.
          * Must not produce NaN or all-zero weights when every particle is far
            off. Multiplying many tiny likelihoods underflows to 0.0; think
            about working in log space.
        """
        raise NotImplementedError

    def resample(self) -> None:
        """Draw N particles with replacement, in proportion to their weights.

        Afterwards weights are uniform again. Any correct scheme passes the
        tests; the course's resampling wheel or low-variance (systematic)
        resampling are both good choices. Return copies, not views.
        """
        raise NotImplementedError

    def estimate(self) -> tuple[float, float, float]:
        """Weighted mean pose (x, y, theta).

        Heading needs a circular mean: averaging 3.1 and -3.1 should give
        about pi, not 0.
        """
        raise NotImplementedError
