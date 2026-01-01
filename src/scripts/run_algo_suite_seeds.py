#!/usr/bin/env python3
import os
import sys
import time
import subprocess

ALGO_LIST = ["DQN", "DoubleDQN", "DuelingDQN", "KFDQN"]
BASE_SEED = 42
ROUNDS = 3
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
