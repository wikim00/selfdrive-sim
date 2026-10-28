"""Run one episode: plan a route, then drive it with noisy motion and sensors.

    python -m selfdrive.run                 # live window
    python -m selfdrive.run --gif demo.gif  # save an animation
    python -m selfdrive.run --headless      # just print the scorecard

Each of your four components is used if it's implemented, and swapped for a
stand-in (see fallbacks.py) if it raises NotImplementedError. Any *other*
exception is a bug in your code and is allowed to crash the run.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import fallbacks
from .algos import control, localization, planning, smoothing
from .car import SimVehicle
from .config import SimConfig
from .world import World, inflate

COMPONENTS = ("planning", "smoothing", "control", "localization")
MAX_DRAWN_PARTICLES = 300


@dataclass
class Frame:
    true_pose: tuple[float, float, float]
    est_pose: tuple[float, float, float]
    particles: np.ndarray | None
    seen: list[int]
    steer: float
    tracking_error: float  # true distance from the path (scoring metric)
    localization_error: float


@dataclass
class RunResult:
    world: World
    raw_path: list[tuple[float, float]]
    path: list[tuple[float, float]]
    frames: list[Frame] = field(default_factory=list)
    status: dict[str, str] = field(default_factory=dict)
    outcome: str = "timeout"  # goal | crash | timeout | no_path

    def metrics(self) -> dict[str, float]:
        if not self.frames:
            return {"steps": 0}
        track = np.array([f.tracking_error for f in self.frames])
        loc = np.array([f.localization_error for f in self.frames])
        return {
            "steps": len(self.frames),
            "track_mean": float(track.mean()),
            "track_max": float(track.max()),
            "track_mse": float((track**2).mean()),
            "loc_mean": float(loc.mean()),
            "loc_max": float(loc.max()),
        }


def _fallback(status: dict, name: str, cfg: SimConfig, what: str) -> None:
    why = "forced" if name in cfg.force_fallback else "not implemented yet"
    status[name] = f"stand-in: {what} ({why})"


# ----------------------------------------------------------------- planning


def plan_route(world: World, cfg: SimConfig, status: dict):
    grid = inflate(world.grid, cfg.margin)
    cells, used_yours = None, False
    if "planning" not in cfg.force_fallback:
        try:
            cells = planning.astar(grid, world.start, world.goal)
            used_yours = True
            status["planning"] = "yours (A*)"
        except NotImplementedError:
            pass
    if not used_yours:
        cells = fallbacks.bfs_plan(grid, world.start, world.goal)
        _fallback(status, "planning", cfg, "breadth-first search")
    if not cells:
        return None, None

    raw = [World.cell_center(tuple(c)) for c in cells]
    smoothed = raw
    if "smoothing" not in cfg.force_fallback:
        try:
            smoothed = [tuple(map(float, p)) for p in smoothing.smooth(raw, cfg.weight_data, cfg.weight_smooth)]
            status["smoothing"] = "yours (gradient descent)"
        except NotImplementedError:
            pass
    if "smoothing" not in status:
        _fallback(status, "smoothing", cfg, "raw grid path")
    return raw, smoothed


# ----------------------------------------------------------------- adapters


class Steering:
    """Uses your CTE + PID, or pure pursuit if either piece isn't there yet."""

    def __init__(self, path, cfg: SimConfig, wheelbase: float, status: dict):
        self.path, self.cfg, self.status = path, cfg, status
        self.index = 0
        self.pp = fallbacks.PurePursuit(path, wheelbase)
        self.pid = None
        if "control" not in cfg.force_fallback:
            try:
                self.pid = control.PIDController(cfg.kp, cfg.ki, cfg.kd)
                status["control"] = f"yours (PID kp={cfg.kp} ki={cfg.ki} kd={cfg.kd})"
            except NotImplementedError:
                pass
        if self.pid is None:
            _fallback(status, "control", cfg, "pure pursuit")

    def steer(self, pose) -> float:
        if self.pid is not None:
            try:
                cte, self.index = control.cross_track_error(self.path, pose[0], pose[1], self.index)
                return float(self.pid.step(cte))
            except NotImplementedError:
                self.pid = None
                _fallback(self.status, "control", self.cfg, "pure pursuit")
        return self.pp.steer(pose)


class Localizer:
    """Uses your particle filter, or ground truth if any method isn't there yet."""

    def __init__(self, world: World, vehicle: SimVehicle, cfg: SimConfig, rng, status: dict):
        self.vehicle, self.cfg, self.status = vehicle, cfg, status
        self.truth = fallbacks.TruthLocalizer(vehicle)
        self.pf = None
        if "localization" not in cfg.force_fallback:
            try:
                n = cfg.noise
                self.pf = localization.ParticleFilter(
                    world.landmarks, cfg.n_particles, vehicle.true_pose, cfg.init_std,
                    n.steering, n.distance, n.range, n.bearing, rng=rng, wheelbase=vehicle.wheelbase,
                )
                status["localization"] = f"yours (particle filter, {cfg.n_particles} particles)"
            except NotImplementedError:
                pass
        if self.pf is None:
            self._give_up()

    def _give_up(self):
        self.pf = None
        _fallback(self.status, "localization", self.cfg, "ground truth from the simulator")

    def _call(self, fn, *args):
        if self.pf is None:
            return None
        try:
            return fn(*args)
        except NotImplementedError:
            self._give_up()
            return None

    def predict(self, steer, distance):
        if self.pf is not None:
            self._call(self.pf.predict, steer, distance)

    def update(self, measurements):
        if self.pf is not None and measurements:
            self._call(self.pf.update, measurements)
            if self.pf is not None:
                self._call(self.pf.resample)

    def estimate(self):
        if self.pf is not None:
            est = self._call(self.pf.estimate)
            if est is not None:
                return tuple(float(v) for v in est)
        return self.truth.estimate()

    def particles(self):
        if self.pf is None:
            return None
        p = self.pf.particles
        step = max(1, len(p) // MAX_DRAWN_PARTICLES)
        return np.array(p[::step], dtype=float, copy=True)


# ----------------------------------------------------------------- episode


def run(cfg: SimConfig) -> RunResult:
    world = World.from_file(cfg.map_path)
    rng = np.random.default_rng(cfg.seed)
    status: dict[str, str] = {}

    raw, path = plan_route(world, cfg, status)
    if path is None:
        return RunResult(world, [], [], status=status, outcome="no_path")
    result = RunResult(world, raw, path, status=status)

    # Start on the first waypoint, facing the second.
    (x0, y0), (x1, y1) = raw[0], raw[min(1, len(raw) - 1)]
    vehicle = SimVehicle(
        world, (x0, y0, math.atan2(y1 - y0, x1 - x0)), cfg.noise, rng,
        max_steer=cfg.max_steer, sensor_range=cfg.sensor_range,
    )
    steering = Steering(path, cfg, vehicle.wheelbase, status)
    localizer = Localizer(world, vehicle, cfg, rng, status)
    goal_xy = World.cell_center(world.goal)

    est = localizer.estimate()
    for _ in range(cfg.max_steps):
        steer = float(np.clip(steering.steer(est), -cfg.max_steer, cfg.max_steer))
        vehicle.drive(steer, cfg.speed)
        localizer.predict(steer, cfg.speed)
        seen = vehicle.sense()
        localizer.update(seen)
        est = localizer.estimate()

        tx, ty, _ = vehicle.true_pose
        result.frames.append(
            Frame(
                true_pose=vehicle.true_pose,
                est_pose=est,
                particles=localizer.particles(),
                seen=[m.landmark_id for m in seen],
                steer=steer,
                tracking_error=fallbacks.path_distance(path, tx, ty),
                localization_error=math.hypot(est[0] - tx, est[1] - ty),
            )
        )
        if not world.is_free(tx, ty):
            result.outcome = "crash"
            break
        if math.hypot(tx - goal_xy[0], ty - goal_xy[1]) < cfg.goal_tolerance:
            result.outcome = "goal"
            break
    return result


def print_report(result: RunResult) -> None:
    print("\nComponent      Status")
    print("-------------  " + "-" * 50)
    for name in COMPONENTS:
        print(f"{name:<13}  {result.status.get(name, '-')}")
    m = result.metrics()
    outcome = {
        "goal": f"reached the goal in {m['steps']} steps",
        "crash": f"CRASHED into a wall after {m['steps']} steps",
        "timeout": f"timed out after {m['steps']} steps without reaching the goal",
        "no_path": "no path from start to goal",
    }[result.outcome]
    print(f"\nOutcome: {outcome}")
    if m["steps"]:
        print(f"Path tracking error:  mean {m['track_mean']:.3f} m   max {m['track_max']:.3f} m")
        print(f"Localization error:   mean {m['loc_mean']:.3f} m   max {m['loc_max']:.3f} m")
    print()


def main(argv=None) -> RunResult:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", type=Path, default=SimConfig.map_path, help="map file (see selfdrive/maps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gif", type=Path, help="save an animation to this file")
    ap.add_argument("--png", type=Path, help="save a summary image of the run")
    ap.add_argument("--headless", action="store_true", help="no window, just the scorecard")
    ap.add_argument("--fallback", default="", help=f"comma list from {','.join(COMPONENTS)} to force stand-ins")
    ap.add_argument("--particles", type=int, default=SimConfig.n_particles)
    ap.add_argument("--every", type=int, default=2, help="animate every Nth step")
    args = ap.parse_args(argv)

    forced = frozenset(s.strip() for s in args.fallback.split(",") if s.strip())
    unknown = forced - set(COMPONENTS)
    if unknown:
        ap.error(f"unknown component(s): {', '.join(sorted(unknown))}")
    cfg = replace(SimConfig(), map_path=args.map, seed=args.seed, force_fallback=forced, n_particles=args.particles)

    result = run(cfg)
    print_report(result)

    if result.frames and (args.gif or args.png or not args.headless):
        if args.gif or args.png or args.headless:
            import matplotlib

            matplotlib.use("Agg")
        from . import viz

        if args.png:
            viz.save_png(result, args.png)
            print(f"Saved {args.png}")
        if args.gif:
            viz.save_gif(result, args.gif, every=args.every)
            print(f"Saved {args.gif}")
        if not (args.gif or args.png or args.headless):
            viz.show(result, every=args.every)
    return result


if __name__ == "__main__":
    main()
