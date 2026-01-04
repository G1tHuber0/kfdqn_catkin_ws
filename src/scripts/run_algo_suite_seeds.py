#!/usr/bin/env python3
import os
import sys
import time
import subprocess

# ALGO_LIST = ["DQN", "DoubleDQN", "DuelingDQN", "KFDQN"]
ALGO_LIST = ["KFDQN"]
BASE_SEED = 66
ROUNDS = 5
SLEEP_SECONDS = 2


def main() -> None:
    seeds = [BASE_SEED + i for i in range(ROUNDS)]
    total_runs = len(seeds) * len(ALGO_LIST)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    run_idx = 0
    for seed in seeds:
        for algo in ALGO_LIST:
            run_idx += 1
            env = os.environ.copy()
            env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 或 ":16:8"
            env.setdefault("TORCH_DETERMINISTIC", "1")            # 如果你用 seeding.py 通过该变量开启确定性
            # env.setdefault("PYTHONHASHSEED", env["SEED"])         # 可选：让 hash 也跟 seed 走
            env.setdefault("OMP_NUM_THREADS", "1")                # 可选：减少并行调度差异
            env.setdefault("MKL_NUM_THREADS", "1")                # 可选
            env["ALGO_NAME"] = algo
            env["SEED"] = str(seed)

            cmd = [sys.executable, "train_env2.py"]

            print("=" * 60, flush=True)
            print(f"Run {run_idx}/{total_runs} | algo={algo} seed={seed}", flush=True)
            print(f"Command: {' '.join(cmd)}", flush=True)
            print("=" * 60, flush=True)

            subprocess.run(cmd, env=env, check=True, cwd=script_dir)

            if SLEEP_SECONDS > 0 and run_idx < total_runs:
                print(f"Cooldown: sleeping {SLEEP_SECONDS}s", flush=True)
                time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
