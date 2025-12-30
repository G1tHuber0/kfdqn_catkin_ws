#!/usr/bin/env python3
import os
import sys
import time
import datetime
import csv
from collections import deque

import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# === 导入项目模块 ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs_ros import env_train  # noqa: F401
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
# ALGO_NAME = "DoubleDQN"
# ALGO_NAME = "DuelingDQN"
ALGO_NAME = "KFDQN"

ENV_NAME = "ObstacleAvoidTrain-v0"
RENDER_MODE = None

CONTINUE_ON_SUCCESS = False

MAX_EPISODES = 1000
MAX_TRAIN_STEPS = 99999999
MAX_EPISODE_STEPS = 100

# [自定义] 
SUCCESS_WINDOW_SIZE = 100 # 成功率统计窗口大小
COLLISION_WINDOW_SIZE = 100  # 碰撞率统计窗口

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

    # 初始化统计队列
    success_window = deque(maxlen=SUCCESS_WINDOW_SIZE)
    collision_window = deque(maxlen=COLLISION_WINDOW_SIZE) 

    csv_path = os.path.join(DATA_DIR, "training_log.csv")
    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    # [修改] CSV头增加 Consistency
    csv_writer.writerow(["Episode", "Total_Steps", "Reward", "Ep_Steps", "Epsilon", "Avg_Loss", "Success", "Collision", "Consistency"])

    print(f"{'='*40}")
    print(f"   Start Training: {ALGO_NAME} (Env2Train)")
    print(f"   Device:         {cfg.device}")
    print(f"   Environment:    {ENV_NAME}")
    print(f"   Output Dir:     {OUTPUT_DIR}")
    print(f"   Train data:     tensorboard --logdir=src/scripts/outputs")
    print(f"   Stop:           steps>={MAX_TRAIN_STEPS} or ep>={MAX_EPISODES}")
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

        # [KFDQN统计] 初始化单回合统计变量
        kfdqn_stats = {
            "hybrid_steps": 0,      # 执行混合策略的总步数
            "consistent_steps": 0   # 混合结果与Q网络一致的步数
        }

        while not (done or truncated):
            if total_steps >= MAX_TRAIN_STEPS:
                break
            
            if ALGO_NAME == "KFDQN":
                kfdqn_m, kfdqn_n = agent.update_parameters(i_episode, current_steps=total_steps)

            action_result = agent.take_action(state, total_steps)
            
            # 解析动作返回结果
            action = 0
            strategy_tag = "unknown"
            q_choice = None # Q网络原本想选的动作

            if isinstance(action_result, tuple):
                action = action_result[0]
                if len(action_result) >= 2: strategy_tag = action_result[1]
                if len(action_result) >= 3: q_choice = action_result[2]
            else:
                action = action_result

            # [KFDQN统计] 统计一致性
            # 只统计 'hya' (Hybrid Action) 阶段，排除 'a_f' (强制模糊) 和 'eps' (随机探索)
            if ALGO_NAME == "KFDQN" and strategy_tag == 'hya':
                kfdqn_stats["hybrid_steps"] += 1
                # 如果最终混合动作 == Q网络想选的动作，视为一致
                if q_choice is not None and action == q_choice:
                    kfdqn_stats["consistent_steps"] += 1

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
                
                # 区分算法更新接口
                if ALGO_NAME == "KFDQN":
                    loss_info = agent.update(transition_dict, episode_idx=i_episode)
                else:
                    loss_info = agent.update(transition_dict)

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

        # 统计滑动窗口
        success_window.append(is_success)
        avg_success_rate = sum(success_window) / len(success_window)
        collision_window.append(is_collision)
        avg_collision_rate = sum(collision_window) / len(collision_window)

        # [KFDQN统计] 计算本回合的一致性比率
        consistency_rate = 0.0
        if kfdqn_stats["hybrid_steps"] > 0:
            consistency_rate = kfdqn_stats["consistent_steps"] / kfdqn_stats["hybrid_steps"]

        writer.add_scalar("Episode/01-Reward", ep_reward, i_episode)
        writer.add_scalar("Episode/06-Steps", ep_steps, i_episode)
        writer.add_scalar("Episode/05-Epsilon", agent.epsilon, i_episode)
        writer.add_scalar("Episode/04-Avg_Loss", avg_loss, i_episode)
        writer.add_scalar(f"Episode/02-SuccessRate_Last{SUCCESS_WINDOW_SIZE}", avg_success_rate, i_episode)
        writer.add_scalar(f"Episode/03-CollisionRate_Last{COLLISION_WINDOW_SIZE}", avg_collision_rate, i_episode)

        if ALGO_NAME == "KFDQN":
            # 记录一致性比率
            writer.add_scalar("KFDQN/00-ActionConsistency", consistency_rate, i_episode)
            writer.add_scalar("KFDQN/01-HybridWeight_m", kfdqn_m, i_episode)
            writer.add_scalar("KFDQN/02-HybridWeight_n", kfdqn_n, i_episode)
            if hasattr(agent, "h1"):
                writer.add_scalar("KFDQN/03-HybridWeight_h1", agent.h1, i_episode)
            if hasattr(agent, "h2"):
                writer.add_scalar("KFDQN/04-HybridWeight_h2", agent.h2, i_episode)

        csv_writer.writerow([i_episode, total_steps, ep_reward, ep_steps, agent.epsilon, avg_loss, is_success, is_collision, consistency_rate])
        csv_file.flush()

        log_str = (
            f"Ep {i_episode:<4} || "
            f"R: {ep_reward:>6.2f} | "
            f"Step: {ep_steps:>3} | "
            f"Loss: {avg_loss:>5.3f} | "
            f"SR: {avg_success_rate:>4.2f} | "
            f"End: {bool(is_success)!s:<5} "
        )
        if ALGO_NAME == "KFDQN":
            log_str += f"| Cons: {consistency_rate:.2f}"
            
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