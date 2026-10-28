"""All the knobs in one place. Command-line flags in run.py override these."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

MAPS_DIR = Path(__file__).parent / "maps"
DEFAULT_MAP = MAPS_DIR / "default.txt"


@dataclass
class NoiseConfig:
    steering: float = 0.02  # rad, std of error added to commanded steering
    distance: float = 0.05  # fraction, std of error in distance actually driven
    range: float = 0.15  # m, std of range measurement error
    bearing: float = 0.04  # rad, std of bearing measurement error


@dataclass
class SimConfig:
    map_path: Path = DEFAULT_MAP
    seed: int = 0

    # Car
    speed: float = 0.25  # meters per step
    max_steer: float = 0.5  # rad
    sensor_range: float = 7.0  # m
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    # Planning and smoothing
    margin: int = 1  # cells of clearance kept from walls when planning
    weight_data: float = 0.5
    weight_smooth: float = 0.2

    # Control (steering = -(kp*cte + ki*integral + kd*derivative))
    kp: float = 0.8
    ki: float = 0.002
    kd: float = 6.0

    # Localization
    n_particles: int = 500
    init_std: tuple[float, float, float] = (0.3, 0.3, 0.1)

    # Episode
    max_steps: int = 1500
    goal_tolerance: float = 0.6  # m

    # Components to replace with the built-in stand-in even if you've
    # implemented them, e.g. {"localization"} to debug your controller alone.
    force_fallback: frozenset[str] = frozenset()
