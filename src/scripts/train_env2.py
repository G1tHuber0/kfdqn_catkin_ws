#!/usr/bin/env python3
import os
import sys
import time
import datetime
import csv
from collections import deque  # [新增] 引入双端队列用于滑动窗口

import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# === 导入项目模块 ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs_ros import env_train  # noqa: F401  (注册训练环境)
from config import Config
from agents.kfdqn_agent import KFDQNAgent
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from utils.replay_buffer import ReplayBuffer

# ==========================================
# 1. 全局配置与参数
# ==========================================
# ALGO_NAME = "DQN"
ALGO_NAME = "DoubleDQN"
# ALGO_NAME = "DuelingDQN"
# ALGO_NAME = "KFDQN"
ENV_NAME = "ObstacleAvoidTrain-v0"
RENDER_MODE = None

CONTINUE_ON_SUCCESS = False

MAX_EPISODES = 1000
MAX_TRAIN_STEPS = 99999999
MAX_EPISODE_STEPS = 200

# [自定义] 成功率统计窗口大小
SUCCESS_WINDOW_SIZE = 10 

CHECKPOINT_STEPS = [2000, 5000, 10000, 20000, 30000, 50000, 75000, 100000, 150000]

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, f"outputs/{ENV_NAME}", f"{ALGO_NAME}_{ENV_NAME}_{TIMESTAMP}")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def main() -> None:
    cfg = Config(algo=ALGO_NAME, env_name=ENV_NAME)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_global = cfg.seed + 99
    env = gym.make(
        ENV_NAME,
        render_mode=RENDER_MODE,
        max_steps=MAX_EPISODE_STEPS,
        continue_on_success=CONTINUE_ON_SUCCESS,
    )

    np.random.seed(seed_global)
    torch.manual_seed(seed_global)

    if ALGO_NAME == "KFDQN":
        agent = KFDQNAgent(cfg)
    elif ALGO_NAME == "DoubleDQN":
        agent = DoubleDQNAgent(cfg)
    elif ALGO_NAME == "DuelingDQN":
        agent = DuelingDQNAgent(cfg)
    else:
        agent = DQNAgent(cfg)
    agent.train_mode()

    replay_buffer = ReplayBuffer(cfg.buffer_size)
    writer = SummaryWriter(log_dir=LOG_DIR)

    # [新增] 初始化成功率统计队列
    success_window = deque(maxlen=SUCCESS_WINDOW_SIZE)

    csv_path = os.path.join(DATA_DIR, "training_log.csv")
    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Episode", "Total_Steps", "Reward", "Ep_Steps", "Epsilon", "Avg_Loss", "Success", "Collision"])

    print(f"{'='*40}")
    print(f"   Start Training: {ALGO_NAME} (Env2Train)")
    print(f"   Environment:    {ENV_NAME}")
    print(f"   Output Dir:     {OUTPUT_DIR}")
    print(f"   Stop: steps>={MAX_TRAIN_STEPS} or ep>={MAX_EPISODES}")
    print(f"{'='*40}\n")

    total_steps = 0
    start_time = time.time()

    pbar = tqdm(
        range(1, MAX_EPISODES + 1),
        desc="Training",
        unit="ep",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
    )

    for i_episode in pbar:
        if total_steps >= MAX_TRAIN_STEPS:
            break

        state, _ = env.reset(seed=seed_global + i_episode)
        ep_reward = 0.0
        ep_steps = 0
        ep_losses: list[float] = []
        done = False
        truncated = False
        info = {}

        while not (done or truncated):
            if total_steps >= MAX_TRAIN_STEPS:
                break

            # agent.update_parameters(i_episode, current_steps=total_steps)

            action_result = agent.take_action(state, total_steps)
            if isinstance(action_result, tuple):
                action = action_result[0]
            else:
                action = action_result

            next_state, reward, done, truncated, info = env.step(action)

            real_done = done and not truncated
            replay_buffer.add(state, action, reward, next_state, real_done)

            state = next_state
            ep_reward += float(reward)
            ep_steps += 1
            total_steps += 1

            if replay_buffer.size() > cfg.minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(cfg.batch_size)
                transition_dict = {
                    "states": b_s,
                    "actions": b_a,
                    "next_states": b_ns,
                    "rewards": b_r,
                    "dones": b_d,
                }
                loss_info = agent.update(transition_dict, episode_idx=i_episode)

                current_loss = 0.0
                if isinstance(loss_info, dict):
                    current_loss = float(loss_info.get("q_loss", 0.0) + loss_info.get("fuzzy_loss", 0.0))
                else:
                    current_loss = float(loss_info)

                ep_losses.append(current_loss)

                if total_steps % 100 == 0:
                    writer.add_scalar("Step/Loss", current_loss, total_steps)

            if total_steps in CHECKPOINT_STEPS:
                save_name = f"{ALGO_NAME}_{TIMESTAMP}_{total_steps}.pth"
                save_path = os.path.join(MODEL_DIR, save_name)
                agent.save(save_path)
                tqdm.write(f">>> [Checkpoint] Model saved: {save_name} at step {total_steps}")

            if total_steps % 10 == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = total_steps / (elapsed_time + 1e-9)
                pbar.set_postfix({"T_Steps": total_steps, "Step/s": f"{steps_per_sec:.1f}"})

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        is_success = 1 if info.get("is_success", False) else 0
        is_collision = 1 if info.get("is_collision", False) else 0

        # [修改] 计算滑动窗口成功率
        success_window.append(is_success)
        avg_success_rate = sum(success_window) / len(success_window)

        writer.add_scalar("Episode/01-Reward", ep_reward, i_episode)
        writer.add_scalar("Episode/05-Steps", ep_steps, i_episode)
        writer.add_scalar("Episode/02-Epsilon", agent.epsilon, i_episode)
        writer.add_scalar("Episode/04-Avg_Loss", avg_loss, i_episode)
        # [修改] 记录最近 N 回合的平均成功率
        writer.add_scalar(f"Episode/03-SuccessRate_Last{SUCCESS_WINDOW_SIZE}", avg_success_rate, i_episode)
        writer.add_scalar("Episode/07-Collision", is_collision, i_episode)

        if hasattr(agent, "m"):
            writer.add_scalar("Episode/06-HybridWeight_m", agent.m, i_episode)

        csv_writer.writerow([i_episode, total_steps, ep_reward, ep_steps, agent.epsilon, avg_loss, is_success, is_collision])
        csv_file.flush()

        log_str = (
            f"Ep {i_episode:<4} || "
            f"R: {ep_reward:>6.2f} | "
            f"Steps: {ep_steps:>4} | "
            f"Loss: {avg_loss:>6.3f} | "
            f"Eps: {agent.epsilon:.3f} | "
            f"SR_{SUCCESS_WINDOW_SIZE}: {avg_success_rate:>4.2f} | "  # [新增] 终端显示滑动成功率
            f"End: {bool(is_success)!s:<5} | "
        )
        tqdm.write(log_str)

        if total_steps >= MAX_TRAIN_STEPS:
            break

    final_save_path = os.path.join(MODEL_DIR, f"{ALGO_NAME}_{TIMESTAMP}_final.pth")
    agent.save(final_save_path)
    print(f"\nTraining Finished. Final model saved to: {final_save_path}")

    env.close()
    csv_file.close()
    writer.close()


if __name__ == "__main__":
    main()