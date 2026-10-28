"""Exercise 3: cross-track error and a PID steering controller.

Course tie-in: the PID control lesson (P, PD, PID, systematic bias).
Until both pieces are implemented the car steers with pure pursuit instead.
"""
from __future__ import annotations

Point = tuple[float, float]


def cross_track_error(path: list[Point], x: float, y: float, index: int = 0) -> tuple[float, int]:
    """Signed distance from the car to the path, and the segment it's on.

    The path is a polyline; segment i runs from path[i] to path[i + 1].

    Starting from segment `index`, project the car onto the current segment.
    While the projection lands past the end of the segment (and there is a
    next segment), move on to the next one. Never move backward.

    Returns:
        (cte, index) where
          * cte is the perpendicular distance from (x, y) to the line through
            the current segment, POSITIVE when the car is to the LEFT of the
            direction of travel and negative to the right;
          * index is the segment used (pass it back in next step).

    Why the sign matters: positive steering turns left, so a car sitting left
    of the path (cte > 0) needs negative steering. The PID below handles that.
    """
    raise NotImplementedError


class PIDController:
    """steering = -(kp * cte + ki * integral + kd * derivative)

    Definitions for each call to step(cte, dt):
        integral   += cte * dt                      (includes the current cte)
        derivative  = (cte - previous_cte) / dt     (0 on the first call after
                                                     construction or reset, so
                                                     there's no derivative kick)
    """

    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()

    def reset(self) -> None:
        """Clear all controller state."""
        raise NotImplementedError

    def step(self, cte: float, dt: float = 1.0) -> float:
        """Return the steering command for this cross-track error."""
        raise NotImplementedError
