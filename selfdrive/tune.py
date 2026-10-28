"""Tune the PID steering gains with your twiddle.

    python -m selfdrive.tune                  # ground-truth pose, fast
    python -m selfdrive.tune --with-pf        # tune through your particle filter

Cost = mean squared distance from the path, averaged over a few seeds, with a
big penalty for crashing or not reaching the goal. Paste the result into
config.py (kp, ki, kd).
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from .algos.tuning import twiddle
from .config import SimConfig
from .run import run

FAIL_PENALTY = 100.0


def make_cost(seeds, with_pf: bool):
    base = SimConfig(force_fallback=frozenset() if with_pf else frozenset({"localization"}))

    def cost(params):
        kp, ki, kd = params
        total = 0.0
        for seed in seeds:
            result = run(replace(base, kp=kp, ki=ki, kd=kd, seed=seed))
            if not result.status.get("control", "").startswith("yours"):
                raise SystemExit("Your PID / cross_track_error aren't implemented yet; finish exercise 3 first.")
            total += result.metrics().get("track_mse", FAIL_PENALTY)
            if result.outcome != "goal":
                total += FAIL_PENALTY
        return total / len(seeds)

    return cost


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-pf", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tolerance", type=float, default=0.01)
    args = ap.parse_args(argv)

    cfg = SimConfig()
    cost = make_cost(range(args.seeds), args.with_pf)
    start = [cfg.kp, cfg.ki, cfg.kd]
    print(f"start  kp={start[0]:.4f} ki={start[1]:.5f} kd={start[2]:.4f}  cost={cost(start):.5f}")
    try:
        best, best_cost = twiddle(cost, start, [0.2, 0.001, 1.0], tolerance=args.tolerance)
    except NotImplementedError:
        raise SystemExit("twiddle isn't implemented yet; that's exercise 4 (selfdrive/algos/tuning.py).")
    print(f"best   kp={best[0]:.4f} ki={best[1]:.5f} kd={best[2]:.4f}  cost={best_cost:.5f}")


if __name__ == "__main__":
    main()
