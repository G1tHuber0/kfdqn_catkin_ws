#!/usr/bin/env python3
import os
import sys
import gymnasium as gym
import numpy as np
import torch
from datetime import datetime
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# --- 1. 路径设置与导入 ---
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../src/kfddpg
parent_dir = os.path.dirname(current_dir)              # .../src

# 将父目录 (src) 加入 python 路径
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# [KF-DDPG] 导入新的 Agent
from agents.kfddpg import KFDDPGAgent
from envs_ros import env  # 注册环境

# --- 2. 超参数配置 ---
ENV_NAME = "Env1"  # 仅针对 Env1 (GoalReach)

if ENV_NAME == "Env1":
    MAX_EPISODES = 600
    MAX_STEPS_PER_EPISODE = 200
else:
    MAX_EPISODES = 6000
    MAX_STEPS_PER_EPISODE = 350

BATCH_SIZE = 256
ACTION_DURATION = 0.2
SEED = 111

# [KF-DDPG] 知识引导参数
# 移除固定衰减，改用论文 Eq.18 自动学习 Theta_G
# ETA_START 等参数已移除

# 模型保存路径
MODEL_DIR = os.path.join(current_dir, "models", f"{ENV_NAME}_KFDDPG") 
os.makedirs(MODEL_DIR, exist_ok=True)

# 日志路径
LOG_DIR = os.path.join(current_dir, "runs", f"{ENV_NAME}_KFDDPG_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def main():
    # --- 3. 初始化环境 ---
    env = gym.make(
        "ENV", 
        max_steps=MAX_STEPS_PER_EPISODE,
        action_duration=ACTION_DURATION,
        robot_model_name="turtlebot3_burger"
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # --- 4. 初始化 KF-DDPG Agent ---
    agent = KFDDPGAgent(state_dim, action_dim)
    print(f"Agent Initialized: KF-DDPG (Theta_G Learnable)")

    # ---  初始化 TensorBoard Writer ---
    writer = SummaryWriter(log_dir=LOG_DIR)
    print(f"TensorBoard log dir: {LOG_DIR}")

    # ---  统计变量 ---
    success_window = deque(maxlen=100) 
    reward_window = deque(maxlen=100)
    total_steps = 0
    best_reward = -float('inf')

    print("Start Training (Knowledge Guided)...")
    
    for episode in range(1, MAX_EPISODES + 1):
        state, info = env.reset(seed=SEED + episode)
        episode_reward = 0
        step_count = 0
        
        # [KF-DDPG] Theta_G 自动学习，不再手动计算 eta
        # eta = ... (Removed)
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # [KF-DDPG] 不传 current_ratio，使用内部学习的 theta_g
            action = agent.select_action(state, noise=True)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, float(terminated))
            agent.update()

            state = next_state
            episode_reward += reward
            step_count += 1
            total_steps += 1

            if done:
                break

        # --- 监控逻辑 ---
        is_success = info.get('is_success', False)
        success_window.append(1 if is_success else 0)
        reward_window.append(episode_reward)

        current_success_rate = np.mean(success_window)
        avg_reward_100 = np.mean(reward_window)

        # 写入 TensorBoard
        writer.add_scalar('Reward/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Reward/Average_Reward_MA100', avg_reward_100, episode)
        writer.add_scalar('Steps/Episode_Steps', step_count, episode)
        writer.add_scalar('Success/Success_Rate_MA100', current_success_rate, episode)
        writer.add_scalar('Steps/Total_Steps', total_steps, episode)
        # 获取当前 Theta_G 值用于记录
        current_theta_g = agent._theta_g().item()
        writer.add_scalar('Param/Theta_G', current_theta_g, episode) # 记录 Theta_G 变化

        print(f"Ep: {episode} | "
              f"Steps: {step_count} | "
              f"Rew: {episode_reward:.1f} | "
              f"AvgRew: {avg_reward_100:.1f} | "
              f"Succ: {current_success_rate:.2f} | "
              f"ThG: {current_theta_g:.2f} | " # 打印当前 Theta_G
              f"Buff: {len(agent.memory)}")
        
        # 保存模型
        if avg_reward_100 > best_reward and episode > 50:
            best_reward = avg_reward_100
            save_path = os.path.join(MODEL_DIR, "best_model")
            agent.save(save_path)
            
        if episode % 50 == 0:
            save_path = os.path.join(MODEL_DIR, f"checkpoint_{episode}")
            agent.save(save_path)

    writer.close()
    print("Training Finished.")
    env.close()

if __name__ == "__main__":
    main()
