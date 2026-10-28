"""The simulated car: a noisy bicycle model with a landmark range-bearing sensor."""
from __future__ import annotations

import numpy as np

from .config import NoiseConfig
from .interfaces import Vehicle
from .models import WHEELBASE, Measurement, bicycle_step, expected_measurement, wrap_angle
from .world import World


class SimVehicle(Vehicle):
    def __init__(
        self,
        world: World,
        pose: tuple[float, float, float],
        noise: NoiseConfig,
        rng: np.random.Generator,
        max_steer: float = 0.5,
        sensor_range: float = 7.0,
        wheelbase: float = WHEELBASE,
    ):
        self.world = world
        self.x, self.y, self.theta = map(float, pose)
        self.noise = noise
        self.rng = rng
        self.max_steer = max_steer
        self.sensor_range = sensor_range
        self.wheelbase = wheelbase

    @property
    def true_pose(self) -> tuple[float, float, float]:
        """Ground truth. Only the simulator knows this; your code shouldn't use it."""
        return (self.x, self.y, self.theta)

    def drive(self, steer: float, distance: float) -> None:
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        actual_steer = steer + self.rng.normal(0.0, self.noise.steering)
        actual_distance = distance * (1.0 + self.rng.normal(0.0, self.noise.distance))
        self.x, self.y, self.theta = bicycle_step(
            self.x, self.y, self.theta, actual_steer, actual_distance, self.wheelbase
        )

    def sense(self) -> list[Measurement]:
        lm = self.world.landmarks
        if len(lm) == 0:
            return []
        ranges, bearings = expected_measurement(self.x, self.y, self.theta, lm[:, 0], lm[:, 1])
        out = []
        for i in np.flatnonzero(ranges <= self.sensor_range):
            r = max(0.0, ranges[i] + self.rng.normal(0.0, self.noise.range))
            b = wrap_angle(bearings[i] + self.rng.normal(0.0, self.noise.bearing))
            out.append(Measurement(int(i), float(r), float(b)))
        return out
