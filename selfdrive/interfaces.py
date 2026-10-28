"""The boundary between the autonomy stack and whatever it is driving.

The run loop only ever talks to a `Vehicle`. Today that's `SimVehicle`.
In the hardware phase you write an `RCVehicle` that implements the same two
methods (send steering/throttle to the servo and ESC; read landmarks from the
camera), and the planner, filter and controller run unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Measurement


class Vehicle(ABC):
    max_steer: float  # radians, symmetric limit

    @abstractmethod
    def drive(self, steer: float, distance: float) -> None:
        """Command the car to steer (radians) and move forward `distance` meters."""

    @abstractmethod
    def sense(self) -> list[Measurement]:
        """Return the landmarks currently visible, with noisy range and bearing."""
