"""Drawing: live window, GIF export, and a one-image run summary."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon

from .run import COMPONENTS, RunResult
from .world import World

BG = "#121418"
FLOOR = "#1b1e24"
WALL = "#4a5160"
TEXT = "#d7dae0"
MUTED = "#8b919c"
RAW_PATH = "#5c6370"
PATH = "#4fc3f7"
TRAIL = "#81c784"
PARTICLE = "#ff8a65"
TRUE_CAR = "#e6e6e6"
EST_CAR = "#ffd54f"
LANDMARK = "#ce93d8"
GOAL = "#81c784"

CAR_LENGTH, CAR_WIDTH = 0.9, 0.5


def _car_corners(x, y, theta):
    c, s = math.cos(theta), math.sin(theta)
    pts = []
    # A body that narrows at the nose so you can see which way it faces.
    for dx, dy in ((0.25, 0.5), (0.6, 0.5), (0.6, -0.5), (0.25, -0.5), (-0.4, -0.5), (-0.4, 0.5)):
        lx = dx * CAR_LENGTH
        ly = dy * CAR_WIDTH * (0.35 if dx == 0.6 else 1.0)
        pts.append((x + lx * c - ly * s, y + lx * s + ly * c))
    return pts


def _status_line(result: RunResult) -> str:
    parts = []
    for name in COMPONENTS:
        mine = result.status.get(name, "").startswith("yours")
        parts.append(f"{name} {'●' if mine else '○'}")
    return "   ".join(parts) + "      ● yours  ○ stand-in"


def _setup(result: RunResult):
    world = result.world
    fig = plt.figure(figsize=(9, 7.4), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.2], hspace=0.22, left=0.06, right=0.97, top=0.93, bottom=0.07)
    ax = fig.add_subplot(gs[0])
    ax_err = fig.add_subplot(gs[1])

    for a in (ax, ax_err):
        a.set_facecolor(FLOOR)
        a.tick_params(colors=MUTED, labelsize=8)
        for spine in a.spines.values():
            spine.set_color(WALL)

    ax.imshow(world.grid, origin="lower", extent=(0, world.cols, 0, world.rows),
              cmap=ListedColormap([FLOOR, WALL]), interpolation="nearest", zorder=0)
    ax.set_xlim(0, world.cols)
    ax.set_ylim(0, world.rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if len(world.landmarks):
        ax.scatter(world.landmarks[:, 0], world.landmarks[:, 1], marker="^", s=70, c=LANDMARK,
                   edgecolors=BG, linewidths=0.8, zorder=3)
    raw = np.array(result.raw_path)
    path = np.array(result.path)
    ax.plot(raw[:, 0], raw[:, 1], ":", color=RAW_PATH, lw=1.2, zorder=1)
    ax.plot(path[:, 0], path[:, 1], "-", color=PATH, lw=1.6, alpha=0.9, zorder=2)
    gx, gy = World.cell_center(world.goal)
    ax.scatter([gx], [gy], marker="*", s=260, c=GOAL, edgecolors=BG, zorder=3)

    fig.text(0.06, 0.965, "Self-driving car simulator", color=TEXT, fontsize=13, weight="bold", va="center")
    fig.text(0.97, 0.965, _status_line(result), color=MUTED, fontsize=8.5, ha="right", va="center")

    n = len(result.frames)
    ax_err.set_xlim(0, max(n, 1))
    top = max(0.5, max((max(f.tracking_error, f.localization_error) for f in result.frames), default=0.5) * 1.1)
    ax_err.set_ylim(0, top)
    ax_err.set_ylabel("error (m)", color=MUTED, fontsize=8)
    ax_err.set_xlabel("step", color=MUTED, fontsize=8)
    ax_err.grid(color=WALL, alpha=0.4, lw=0.5)
    return fig, ax, ax_err


def _artists(ax, ax_err):
    a = {}
    a["rays"] = ax.plot([], [], "-", color=LANDMARK, lw=0.6, alpha=0.35, zorder=2)[0]
    a["trail"] = ax.plot([], [], "-", color=TRAIL, lw=1.4, zorder=4)[0]
    a["particles"] = ax.scatter([], [], s=3, c=PARTICLE, alpha=0.45, linewidths=0, zorder=5)
    a["car"] = Polygon(np.zeros((6, 2)), closed=True, fc=TRUE_CAR, ec=BG, lw=0.8, zorder=7)
    a["est"] = Polygon(np.zeros((6, 2)), closed=True, fill=False, ec=EST_CAR, lw=1.4, zorder=8)
    ax.add_patch(a["car"])
    ax.add_patch(a["est"])
    a["info"] = ax_err.text(0.0, 1.06, "", transform=ax_err.transAxes, color=TEXT, fontsize=8.5,
                            va="bottom", family="monospace")
    a["track"] = ax_err.plot([], [], color=PATH, lw=1.2, label="distance from path")[0]
    a["loc"] = ax_err.plot([], [], color=EST_CAR, lw=1.2, label="localization error")[0]
    leg = ax_err.legend(loc="upper right", fontsize=7.5, frameon=False)
    for t in leg.get_texts():
        t.set_color(MUTED)
    return a


def _draw_frame(result: RunResult, a, i: int):
    frames = result.frames
    f = frames[i]
    tx, ty, tt = f.true_pose
    trail = np.array([fr.true_pose[:2] for fr in frames[: i + 1]])
    a["trail"].set_data(trail[:, 0], trail[:, 1])
    a["car"].set_xy(_car_corners(tx, ty, tt))
    a["est"].set_xy(_car_corners(*f.est_pose))
    if f.particles is not None:
        a["particles"].set_offsets(f.particles[:, :2])
    else:
        a["particles"].set_offsets(np.empty((0, 2)))

    lm = result.world.landmarks
    xs, ys = [], []
    for lid in f.seen:
        xs += [tx, lm[lid, 0], np.nan]
        ys += [ty, lm[lid, 1], np.nan]
    a["rays"].set_data(xs, ys)

    steps = np.arange(1, i + 2)
    a["track"].set_data(steps, [fr.tracking_error for fr in frames[: i + 1]])
    a["loc"].set_data(steps, [fr.localization_error for fr in frames[: i + 1]])

    last = i == len(frames) - 1
    banner = {"goal": "GOAL", "crash": "CRASH", "timeout": "TIMEOUT"}.get(result.outcome, "") if last else ""
    a["info"].set_text(
        f"step {i + 1:4d}   steer {math.degrees(f.steer):+5.1f}°   "
        f"off path {f.tracking_error:.2f} m   loc err {f.localization_error:.2f} m"
        + (f"   {banner}" if banner else "")
    )
    return list(a.values())


def animate(result: RunResult, every: int = 2, fps: int = 25):
    fig, ax, ax_err = _setup(result)
    a = _artists(ax, ax_err)
    idx = list(range(0, len(result.frames), max(1, every)))
    if idx[-1] != len(result.frames) - 1:
        idx.append(len(result.frames) - 1)
    idx += [idx[-1]] * fps  # hold the last frame for a second
    anim = FuncAnimation(fig, lambda k: _draw_frame(result, a, k), frames=idx, interval=1000 / fps, blit=False)
    return fig, anim


def save_gif(result: RunResult, path, every: int = 2, fps: int = 25) -> None:
    fig, anim = animate(result, every, fps)
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def save_png(result: RunResult, path) -> None:
    fig, ax, ax_err = _setup(result)
    a = _artists(ax, ax_err)
    _draw_frame(result, a, len(result.frames) - 1)
    fig.savefig(str(path), dpi=110, facecolor=BG)
    plt.close(fig)


def show(result: RunResult, every: int = 2) -> None:
    fig, anim = animate(result, every)
    plt.show()
