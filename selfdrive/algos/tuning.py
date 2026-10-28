"""Exercise 4: twiddle (coordinate-ascent parameter search).

Course tie-in: twiddle in the PID lesson. Once this and your PID work, run
`python -m selfdrive.tune` to tune the steering gains on the real map.
"""
from __future__ import annotations

from typing import Callable, Sequence


def twiddle(
    cost: Callable[[list[float]], float],
    params: Sequence[float],
    deltas: Sequence[float],
    tolerance: float = 1e-3,
    max_iterations: int = 1000,
) -> tuple[list[float], float]:
    """Minimize `cost(params)` by nudging one parameter at a time.

    For each parameter: try increasing it by its delta. If that helps, keep it
    and grow the delta. If not, try decreasing by the delta. If that helps,
    keep it and grow the delta; otherwise restore it and shrink the delta.
    Stop when sum(deltas) < tolerance or after max_iterations sweeps.

    Returns:
        (best_params, best_cost), where best_cost == cost(best_params).

    Requirements:
        * Don't modify the `params` or `deltas` you were given.
        * Pick sensible grow/shrink factors (the course uses 1.1 and 0.9).

    The sim's cost function is noisy (random seeds), so twiddle can chase
    noise. tune.py averages a few seeds per evaluation to help.
    """
    raise NotImplementedError
