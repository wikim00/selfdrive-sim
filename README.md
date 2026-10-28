# Self-driving car simulator

A small autonomy stack for a car-like robot, built in two phases: a Python
simulator now, a real RC car later. The car plans a route through a map,
smooths it into something drivable, follows it with a steering controller,
and works out where it is from noisy landmark sightings.

```
 map ──► A* ──► smoothing ──► path
                               │
            ┌──────────────────▼───────────────┐
            │  particle filter ──► pose ──► CTE + PID ──► steering ──► car
            │        ▲                                                  │
            └────────┴──────────── landmark range/bearing ◄─────────────┘
```

## Quick start

```bash
pip install -r requirements.txt
python -m selfdrive.run                    # live window
python -m selfdrive.run --gif demo.gif     # save an animation
python -m selfdrive.run --headless         # scorecard only
pytest                                     # see what's left to do
```

It runs from day one. Every component you haven't written yet is replaced by
a stand-in, and the header of the window shows which is which (● yours,
○ stand-in). The stand-ins are deliberately different techniques (BFS,
pure pursuit, reading the true pose from the simulator), so they're something
to compare against rather than an answer key.

## The exercises

All your code lives in `selfdrive/algos/`. Each file has the contract in its
docstrings and a matching test file.

| # | File | What you build | Tests |
|---|------|----------------|-------|
| 1 | `planning.py` | A* on the occupancy grid | `pytest tests/test_planning.py` |
| 2 | `smoothing.py` | Gradient-descent path smoothing | `pytest tests/test_smoothing.py` |
| 3 | `control.py` | Cross-track error + PID steering | `pytest tests/test_control.py` |
| 4 | `tuning.py` | Twiddle, then `python -m selfdrive.tune` | `pytest tests/test_tuning.py` |
| 5 | `localization.py` | Particle filter (predict, update, resample, estimate) | `pytest tests/test_localization.py` |

After each one, run the sim and watch it change. Useful while debugging:

```bash
python -m selfdrive.run --fallback localization   # test your controller on the true pose
python -m selfdrive.run --seed 3                  # different noise
python -m selfdrive.run --particles 100           # how few particles still works?
```

The scorecard reports distance from the planned path and localization error,
measured against ground truth the same way no matter whose code is driving.
The default PID gains are deliberately untuned; exercise 4 is where they get
good.

## What's given

| File | Purpose |
|------|---------|
| `models.py` | Bicycle motion model and range/bearing sensor model, shared by sim and filter |
| `world.py` | ASCII maps (`#` wall, `L` landmark, `S` start, `G` goal), wall inflation |
| `car.py` | Simulated car with steering, distance and sensor noise |
| `interfaces.py` | The `Vehicle` interface the run loop drives (the hardware seam) |
| `fallbacks.py` | Stand-ins and the scoring metric |
| `run.py`, `viz.py`, `tune.py` | Episode loop, drawing, gain tuning |
| `config.py` | Every knob: noise levels, speed, gains, smoothing weights, particles |

New maps go in `selfdrive/maps/` and run with `--map`.

## Academic integrity

This project uses the *ideas* from CS 7638, but it's a separate codebase with
its own interfaces, maps and tests. Write your implementations here from
scratch rather than pasting in graded problem-set code, and don't publish
anything that would amount to assignment solutions. Done that way, it's a
legitimate portfolio project you can explain line by line.

## Phase 2: hardware

The algorithms in `selfdrive/algos/` never import the simulator. To drive a
real car you write one class:

```python
class RCVehicle(Vehicle):
    def drive(self, steer, distance): ...   # servo angle + ESC throttle for `distance`
    def sense(self): ...                     # -> list[Measurement]
```

A practical first build:

* **Chassis and compute:** a 1/16 or 1/10 RC car with a Raspberry Pi and a
  PWM driver board for the steering servo and ESC (the Donkey Car community
  documents this setup well).
* **Landmarks:** printed ArUco markers on posts. OpenCV's marker pose
  estimation gives range and bearing directly, so your particle filter's
  `update` works unchanged.
* **Odometry:** a wheel encoder (or a fixed, calibrated speed to start) feeds
  `predict`.
* **Map:** tape out the grid on a garage floor and write it as a map file.
* **Calibration:** measure the wheelbase, the real steering limit and the
  noise levels, then update `models.py` and `config.py`. Expect to retune PID.

## Stretch goals

* Global localization: start the particles spread over the whole map (the
  "kidnapped robot" problem).
* An extended Kalman filter as a second localizer, compared against the
  particle filter on the same runs.
* GraphSLAM: drive with unknown landmark positions and build the map.
* Obstacle-aware smoothing, so the smoother stops relying on wall inflation.
* Replanning when a new obstacle appears mid-run.
