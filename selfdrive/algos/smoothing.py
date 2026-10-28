"""Exercise 2: path smoothing.

Course tie-in: the path smoothing lesson (gradient descent on two competing
objectives). Until this is implemented the car drives the raw grid path,
corners and all.
"""
from __future__ import annotations

Point = tuple[float, float]


def smooth(
    path: list[Point],
    weight_data: float = 0.5,
    weight_smooth: float = 0.1,
    tolerance: float = 1e-6,
) -> list[Point]:
    """Smooth a polyline while keeping it near the original.

    Two pulls act on every interior point:
      * weight_data pulls it back toward its original position.
      * weight_smooth pulls it toward the midpoint of its neighbors.
    Iterate until the total change in one full sweep is below `tolerance`.

    Args:
        path: list of (x, y) points (cell centers from the planner).

    Returns:
        A new list of (x, y) floats, same length as `path`.

    Requirements:
        * The first and last points never move.
        * Don't modify the input list.
        * weight_smooth = 0 returns the original points.
        * A straight, evenly spaced line comes back unchanged.

    Things to think about: what happens if weight_data is 0? Why does this
    smoothing ignore obstacles, and how does the planner's wall margin
    (config.margin) compensate for that?
    """
    raise NotImplementedError
