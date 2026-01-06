#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a fixed list of goal positions, matching env_train.py sampling constraints.
Default output: src/scripts/envs_ros/list_goal.csv

Also optionally plot the goal point cloud.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Plot is optional; import lazily to avoid hard dependency when --plot not used.


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    r: float


def sample_goal(
    rng: np.random.Generator,
    *,
    robot_x: float,
    robot_y: float,
    lim: float,
    goal_d_min: float,
    goal_d_max: float,
    obstacles: List[Obstacle],
    obstacle_margin: float,
    max_tries: int = 1000,
) -> Tuple[float, float]:
    last_gx = last_gy = 0.0
    for _ in range(max_tries):
        gx = float(rng.uniform(-lim, lim))
        gy = float(rng.uniform(-lim, lim))
        last_gx, last_gy = gx, gy

        d = float(math.hypot(gx - robot_x, gy - robot_y))
        if not (goal_d_min <= d <= goal_d_max):
            continue

        ok = True
        for obs in obstacles:
            if math.hypot(gx - obs.x, gy - obs.y) < (obs.r + obstacle_margin):
                ok = False
                break

        if ok:
            return gx, gy

    return last_gx, last_gy


def read_goals_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["goal_x"]))
            ys.append(float(row["goal_y"]))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def plot_goals(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    out_path: Path,
    lim: float,
    robot_x: float,
    robot_y: float,
    goal_d_min: float,
    goal_d_max: float,
    obstacles: List[Obstacle],
    obstacle_margin: float,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Points
    ax.scatter(xs, ys, s=8, alpha=0.7)

    # Map boundary (square)
    ax.add_patch(
        Rectangle(
            (-lim, -lim),
            2 * lim,
            2 * lim,
            fill=False,
            linewidth=1.5,
        )
    )

    # Obstacles (actual radius) + safety margin radius
    for obs in obstacles:
        ax.add_patch(Circle((obs.x, obs.y), obs.r, fill=False, linewidth=1.5))
        ax.add_patch(
            Circle((obs.x, obs.y), obs.r + obstacle_margin, fill=False, linestyle="--", linewidth=1.0)
        )

    # Robot spawn point
    ax.scatter([robot_x], [robot_y], s=60, marker="x")
    # Distance ring [goal_d_min, goal_d_max]
    ax.add_patch(Circle((robot_x, robot_y), goal_d_min, fill=False, linestyle="--", linewidth=1.0))
    ax.add_patch(Circle((robot_x, robot_y), goal_d_max, fill=False, linestyle="--", linewidth=1.0))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim - 0.1, lim + 0.1)
    ax.set_ylim(-lim - 0.1, lim + 0.1)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260106)

    # Mirror env defaults; adjust if your env_train.py differs
    ap.add_argument("--map_xy_limit", type=float, default=2.0)
    ap.add_argument("--wall_margin", type=float, default=0.4)
    ap.add_argument("--goal_d_min", type=float, default=0.5)
    ap.add_argument("--goal_d_max", type=float, default=3.0)
    ap.add_argument("--obstacle_margin", type=float, default=0.2)

    # Robot spawn used by _sample_goal constraints.
    ap.add_argument("--robot_x", type=float, default=0.0)
    ap.add_argument("--robot_y", type=float, default=0.0)

    ap.add_argument(
        "--out",
        type=str,
        default=str(
           Path(__file__).resolve().parents[1] / "envs_ros" / "list_goal.csv"
            ),
    )

    # Plot options
    ap.add_argument("--plot", action="store_true", help="Generate a scatter plot PNG after writing CSV")
    ap.add_argument(
        "--plot_out",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1] / "envs_ros" / "list_goal.png"
            ),
        help="Output path for the plot PNG",
    )

    args = ap.parse_args()

    lim = float(args.map_xy_limit - args.wall_margin)
    rng = np.random.default_rng(args.seed)

    # Mirror env_train.py obstacles list
    obstacles = [
        Obstacle(-0.6, -0.6, 0.35),
        Obstacle(-0.6,  0.6, 0.35),
        Obstacle( 0.6, -0.6, 0.35),
        Obstacle( 0.6,  0.6, 0.35),
        Obstacle( 1.7,  0.0, 0.35),
        Obstacle(-1.7,  0.0, 0.35),
        Obstacle( 0.0,  1.7, 0.35),
        Obstacle( 0.0, -1.7, 0.35),
    ]

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "goal_x", "goal_y"])
        for ep in range(args.n):
            gx, gy = sample_goal(
                rng,
                robot_x=args.robot_x,
                robot_y=args.robot_y,
                lim=lim,
                goal_d_min=args.goal_d_min,
                goal_d_max=args.goal_d_max,
                obstacles=obstacles,
                obstacle_margin=args.obstacle_margin,
            )
            w.writerow([ep, gx, gy])

    print(f"[OK] wrote {args.n} goals to: {out_path}")
    xs, ys = read_goals_csv(out_path)
    plot_path = Path(args.plot_out).resolve()
    plot_goals(
        xs=xs,
        ys=ys,
        out_path=plot_path,
        lim=lim,
        robot_x=float(args.robot_x),
        robot_y=float(args.robot_y),
        goal_d_min=float(args.goal_d_min),
        goal_d_max=float(args.goal_d_max),
        obstacles=obstacles,
        obstacle_margin=float(args.obstacle_margin),
        title=f"Goal Point Cloud (n={args.n}, seed={args.seed})",
    )
    print(f"[OK] wrote plot to: {plot_path}")



if __name__ == "__main__":
    main()
