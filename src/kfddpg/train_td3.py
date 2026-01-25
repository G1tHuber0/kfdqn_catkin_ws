#!/usr/bin/env python3
import os
import sys
import gymnasium as gym
import numpy as np
import torch
from datetime import datetime
from collections import deque  # 用于滑动窗口计算
from torch.utils.tensorboard import SummaryWriter # TensorBoard 写入器

# --- 1. 路径设置与导入 ---
current_dir = os.path.dirname(os.path.abspath(__file__)) # 得到 .../src/kfddpg
parent_dir = os.path.dirname(current_dir)              # 得到 .../src

# 将父目录 (src) 加入 python 路径
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# [修改] 导入 TD3 Agent
from agents.td3 import TD3Agent
from envs_ros import env  # 确保环境已注册

# --- 2. 超参数配置 ---
ENV_NAME = "Env1"  

if ENV_NAME == "Env1":
    MAX_EPISODES = 600
    MAX_STEPS_PER_EPISODE = 200
else:
    MAX_EPISODES = 6000
    MAX_STEPS_PER_EPISODE = 350

BATCH_SIZE = 256
ACTION_DURATION = 0.2
SEED = 123

# 模型保存路径
MODEL_DIR = os.path.join(current_dir, "models", f"{ENV_NAME}_TD3") 
os.makedirs(MODEL_DIR, exist_ok=True)

# [修改] TensorBoard 日志路径 (加上 TD3 前缀区分)
LOG_DIR = os.path.join(current_dir, "runs", f"{ENV_NAME}_TD3_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def main():
    # --- 3. 初始化环境 ---
    env = gym.make(
        "ENV", 
        max_steps=MAX_STEPS_PER_EPISODE,
        action_duration=ACTION_DURATION,
        robot_model_name="turtlebot3_burger",
        # [修改] 强制关闭连续导航，每次到达目标后重置回原点，降低初期难度
        continue_on_success=False 
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # --- 4. 初始化 Agent (TD3) ---
    agent = TD3Agent(state_dim, action_dim)

    # ---  初始化 TensorBoard Writer ---
    writer = SummaryWriter(log_dir=LOG_DIR)
    print(f"TensorBoard log dir: {LOG_DIR}")
    print("Run command: tensorboard --logdir src/kfddpg/runs")

    # ---  统计变量 ---
    success_window = deque(maxlen=100) 
    reward_window = deque(maxlen=100)
    total_steps = 0
    
    best_reward = -float('inf')

    print("Start Training TD3...")
    
    for episode in range(1, MAX_EPISODES + 1):
        state, info = env.reset(seed=SEED + episode)
        episode_reward = 0
        step_count = 0
        
        # TD3 通常不需要显式 reset noise，它是无状态的高斯噪声，但保留也无妨
        # agent.noise.reset() 

        for step in range(MAX_STEPS_PER_EPISODE):
            # TD3 训练时 noise=True (添加探索噪声)
            action = agent.select_action(state, noise=True)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            
            # 这里的 float(terminated) 表示撞墙或到达算结束，超时不算
            agent.memory.push(state, action, reward, next_state, float(terminated))
            
            # --- [修改] TD3 更新逻辑 ---
            # TD3 的 update 返回 (critic_loss, actor_loss)
            # 注意: actor_loss 可能是 None (因为延迟更新)
            c_loss, a_loss = agent.update()
            
            # 记录 Loss 到 TensorBoard (这对 TD3 调试很重要)
            if c_loss is not None:
                writer.add_scalar('Loss/Critic', c_loss, total_steps)
            if a_loss is not None:
                writer.add_scalar('Loss/Actor', a_loss, total_steps)

            state = next_state
            episode_reward += reward
            step_count += 1
            total_steps += 1

            if done:
                break

        # --- 6. 核心监控逻辑 ---
        
        # 1. 判定是否成功
        is_success = info.get('is_success', False)
        success_window.append(1 if is_success else 0)
        
        # 2. 更新奖励窗口
        reward_window.append(episode_reward)

        # 3. 计算统计值
        current_success_rate = np.mean(success_window)
        avg_reward_100 = np.mean(reward_window)       

        # 4. 写入 TensorBoard
        writer.add_scalar('Reward/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Reward/Average_Reward_MA100', avg_reward_100, episode)
        writer.add_scalar('Steps/Episode_Steps', step_count, episode)
        writer.add_scalar('Success/Success_Rate_MA100', current_success_rate, episode)
        writer.add_scalar('Steps/Total_Steps', total_steps, episode)
        
        # 打印日志
        print(f"Ep: {episode} | "
              f"Steps: {step_count} | "
              f"Reward: {episode_reward:.1f} | "
              f"AvgRew: {avg_reward_100:.1f} | "
              f"SuccRate: {current_success_rate:.2f} | "
              f"Buff: {len(agent.memory)} | "
              f"TotalSteps: {total_steps}")
        

        # 保存最优模型
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