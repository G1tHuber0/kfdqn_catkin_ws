#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成一个 1.7x1.7 正方形区域内的随机目标点序列（无障碍物）。
默认输出到当前目录下的 simple_goals.csv
同时，如果指定 --plot，会生成对应的点云图 simple_goals.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np

# 绘图是可选的，因此在需要时才导入 matplotlib
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle


def generate_square_goals(
    rng: np.random.Generator,
    n: int,
    side_length: float = 1.7
) -> list[tuple[int, float, float]]:
    """在中心为(0,0)，边长为 side_length 的正方形内均匀采样"""
    half_side = side_length / 2.0
    goals = []
    
    for i in range(n):
        # rng.uniform(low, high) 产生 [low, high) 之间的均匀分布
        gx = float(rng.uniform(-half_side, half_side))
        gy = float(rng.uniform(-half_side, half_side))
        goals.append((i, gx, gy))
        
    return goals


def plot_simple_goals(
    xs: List[float], 
    ys: List[float], 
    side_length: float, 
    out_path: Path, 
    title: str
) -> None:
    """绘制正方形区域内的目标点云图"""
    # 延迟导入 matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 7))
    half_side = side_length / 2.0

    # 绘制目标点云
    ax.scatter(xs, ys, s=10, alpha=0.6, c='blue', label='Goals')

    # 绘制 1.7x1.7 的正方形边界
    rect = Rectangle(
        (-half_side, -half_side), 
        side_length, side_length, 
        fill=False, color='red', linewidth=2, label='Boundary (1.7x1.7)'
    )
    ax.add_patch(rect)

    ax.set_aspect("equal") # 保持 x, y 轴比例一致，使正方形看起来是正方形
    
    # 稍微扩展一下显示范围，让边界可见
    plot_lim = half_side + 0.1 
    ax.set_xlim(-plot_lim, plot_lim)
    ax.set_ylim(-plot_lim, plot_lim)
    
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle='--', alpha=0.5) # 添加网格线
    ax.legend() # 显示图例

    out_path.parent.mkdir(parents=True, exist_ok=True) # 确保输出目录存在
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Plot saved to: {out_path}")
    plt.close(fig) # 关闭图形，释放内存


def main():
    parser = argparse.ArgumentParser(description="生成 1.7x1.7 无障碍随机目标点序列。")
    parser.add_argument("--n", type=int, default=1000, help="生成目标的数量")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子，用于复现结果")
    parser.add_argument("--side", type=float, default=1.7, help="正方形区域的边长，默认为 1.7m")
    parser.add_argument(
        "--out", 
        type=str, 
        default="simple_goals.csv", 
        help="CSV 文件输出路径"
    )
    parser.add_argument("--plot", action="store_true", help="是否生成可视化图表 (PNG)")
    parser.add_argument(
        "--plot_out",
        type=str,
        default="simple_goals.png", # 默认为CSV同名但后缀为.png
        help="可视化图表的输出路径 (PNG)，若不指定则默认为CSV文件的同名PNG"
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    
    # 1. 生成数据
    goals = generate_square_goals(rng, args.n, args.side)

    # 2. 写入 CSV
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "goal_x", "goal_y"])
        writer.writerows(goals)
    
    print(f"[OK] Wrote {args.n} goals to: {out_path}")

    xs = [g[1] for g in goals]  
    ys = [g[2] for g in goals]
    
    plot_out_path = Path(args.plot_out).resolve() if args.plot_out else out_path.with_suffix(".png")
    
    plot_simple_goals(
        xs, 
        ys, 
        args.side, 
        plot_out_path, 
        title=f"Random Goals (Square {args.side}x{args.side}m, n={args.n})"
    )

if __name__ == "__main__":
    main()