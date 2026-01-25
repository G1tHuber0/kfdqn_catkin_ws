#!/usr/bin/env python3
import os
import sys
import gymnasium as gym
import numpy as np
import torch
from datetime import datetime
from collections import deque  # [新增] 用于滑动窗口计算
from torch.utils.tensorboard import SummaryWriter # [新增] TensorBoard 写入器

# --- 1. 路径设置与导入 ---
current_dir = os.path.dirname(os.path.abspath(__file__)) # 得到 .../src/kfddpg
parent_dir = os.path.dirname(current_dir)              # 得到 .../src

# 将父目录 (src) 加入 python 路径
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from agents.ddpg import DDPGAgent
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
SEED = 111

# 模型保存路径
MODEL_DIR = os.path.join(current_dir, "models", f"{ENV_NAME}_DDPG") 
os.makedirs(MODEL_DIR, exist_ok=True)

# [新增] TensorBoard 日志路径 (通常放在 runs/ 目录下，按时间戳区分)
LOG_DIR = os.path.join(current_dir, "runs", f"{ENV_NAME}_DDPG_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

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
    
    # --- 4. 初始化 Agent ---
    agent = DDPGAgent(state_dim, action_dim)

    # ---  初始化 TensorBoard Writer ---
    writer = SummaryWriter(log_dir=LOG_DIR)
    print(f"TensorBoard log dir: {LOG_DIR}")
    print("Run command: tensorboard --logdir src/runs")

    # ---  统计变量 ---
    # 成功率滑动窗口 (只保留最近 100 次结果，1=成功, 0=失败)
    success_window = deque(maxlen=100) 
    # 奖励滑动窗口 (用于计算平滑后的平均奖励)
    reward_window = deque(maxlen=100)
    # 总步数
    total_steps = 0
    
    best_reward = -float('inf')

    print("Start Training...")
    
    for episode in range(1, MAX_EPISODES + 1):
        state, info = env.reset(seed=SEED + episode)
        episode_reward = 0
        step_count = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state, noise=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated

            # 存入 buffer 时，建议把 truncated 视为未完成 (done=0)，只有 terminated (撞墙/到达) 才是 done=1
            # 但为了保持基础 DDPG 简单，这里暂且存 float(terminated)
            agent.memory.push(state, action, reward, next_state, float(terminated))
            agent.update()

            state = next_state
            episode_reward += reward
            step_count += 1
            total_steps += 1

            if done:
                break

        # --- [新增] 6. 核心监控逻辑 ---
        
        # 1. 判定是否成功 (根据 env.py 返回的 info)
        # env.py 里: info["is_success"] = True (如果到达)
        is_success = info.get('is_success', False)
        success_window.append(1 if is_success else 0)
        
        # 2. 更新奖励窗口
        reward_window.append(episode_reward)

        # 3. 计算统计值
        current_success_rate = np.mean(success_window)  # 最近 100 回合成功率 (0~1)
        avg_reward_100 = np.mean(reward_window)         # 最近 100 回合平均奖励

        # 4. 写入 TensorBoard
        # (a) 平均回合奖励 (Raw 和 Smooth 都可以看，这里记 Raw，TensorBoard 前端可以自己平滑)
        writer.add_scalar('Reward/Episode_Reward', episode_reward, episode)
        # (b) 也可以记一个 100 次平均的奖励曲线，更加平滑
        writer.add_scalar('Reward/Average_Reward_MA100', avg_reward_100, episode)
        # (c) 按回合统计训练步数 (看是否能更快到达)
        writer.add_scalar('Steps/Episode_Steps', step_count, episode)
        # (d) 固定滑动窗口 100 的成功率
        writer.add_scalar('Success/Success_Rate_MA100', current_success_rate, episode)
        # (e) 总步数统计
        writer.add_scalar('Steps/Total_Steps', total_steps, episode)
        # 打印日志
        print(f"Ep: {episode} | "
              f"Steps: {step_count} | "
              f"Reward: {episode_reward:.1f} | "
              f"AvgRew: {avg_reward_100:.1f} | "
              f"SuccRate: {current_success_rate:.2f} | "
              f"Buff: {len(agent.memory)} | "
              f"TotalSteps: {total_steps}")
        

        # 保存最优模型 (使用平滑后的奖励来判断，更稳定)
        if avg_reward_100 > best_reward and episode > 50:
            best_reward = avg_reward_100
            save_path = os.path.join(MODEL_DIR, "best_model")
            agent.save(save_path)
            
        if episode % 50 == 0:
            save_path = os.path.join(MODEL_DIR, f"checkpoint_{episode}")
            agent.save(save_path)

    writer.close() # [新增] 关闭写入器
    print("Training Finished.")
    env.close()

if __name__ == "__main__":
    main()