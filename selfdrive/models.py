"""Shared math: the car's motion model and the landmark sensor model.

These are *given* (not exercises) because the simulator and your particle
filter must agree on the same model. When you move to hardware you keep these
functions and just re-measure WHEELBASE and the noise levels on the real car.

Conventions used everywhere in this project:
  * Units are meters and radians. One map cell is 1 m x 1 m.
  * x points right, y points up, heading theta is measured counter-clockwise
    from +x and kept in [-pi, pi).
  * Positive steering turns the car left (counter-clockwise).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WHEELBASE = 0.6  # meters between front and rear axle


def wrap_angle(a):
    """Wrap an angle (or array of angles) into [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def bicycle_step(x, y, theta, steer, distance, wheelbase: float = WHEELBASE):
    """Advance a kinematic bicycle model by `distance` meters.

    Works on plain floats or on numpy arrays of any matching shape, so you can
    move one car or 1,000 particles with the same call.

    Returns (new_x, new_y, new_theta).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    theta = np.asarray(theta, dtype=float)
    steer = np.asarray(steer, dtype=float)
    distance = np.asarray(distance, dtype=float)

    turn = distance / wheelbase * np.tan(steer)
    straight = np.abs(turn) < 1e-6
    safe_turn = np.where(straight, 1.0, turn)
    radius = distance / safe_turn

    # Turning: rotate around the instantaneous center of curvature.
    cx = x - np.sin(theta) * radius
    cy = y + np.cos(theta) * radius
    new_theta = theta + turn
    arc_x = cx + np.sin(new_theta) * radius
    arc_y = cy - np.cos(new_theta) * radius

    # Going (almost) straight: plain translation.
    line_x = x + distance * np.cos(theta)
    line_y = y + distance * np.sin(theta)

    nx = np.where(straight, line_x, arc_x)
    ny = np.where(straight, line_y, arc_y)
    nt = wrap_angle(np.where(straight, theta, new_theta))

    if nx.ndim == 0:
        return float(nx), float(ny), float(nt)
    return nx, ny, nt


@dataclass(frozen=True)
class Measurement:
    """One landmark sighting: which landmark, how far, and at what angle.

    bearing is relative to the car's heading (0 = straight ahead,
    positive = to the left), in [-pi, pi).
    """

    landmark_id: int
    range: float
    bearing: float


def expected_measurement(x, y, theta, lx, ly):
    """Noise-free (range, bearing) from pose (x, y, theta) to landmark (lx, ly).

    Broadcasts like numpy: pass arrays of particle poses and a single landmark
    to get one expected measurement per particle.
    """
    dx = np.asarray(lx, dtype=float) - np.asarray(x, dtype=float)
    dy = np.asarray(ly, dtype=float) - np.asarray(y, dtype=float)
    rng = np.hypot(dx, dy)
    bearing = wrap_angle(np.arctan2(dy, dx) - np.asarray(theta, dtype=float))
    return rng, bearing
