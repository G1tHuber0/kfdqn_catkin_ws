#!/usr/bin/env python3
import os
import sys
import time
import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm

# 导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import envs_ros.ros_gazebo_mobile_robot_env  # 注册环境
from config import Config
from agents.dqn_agent import DQNAgent
from agents.kfdqn_agent import KFDQNAgent

# ==========================================
# 配置部分
# ==========================================
ALGO_NAME = "KFDQN"           # 算法名称
ENV_ID = "GoalReachEval-v0"   # 使用评估环境 (固定路径)

MODEL_PATH = "src/scripts/outputs/KFDQN_GoalReachTrain-v0_20251226_151643/models/KFDQN_20251226_151643_final.pth"

EVAL_EPISODES = 100           # 测试多少轮
MAX_STEPS = 1000               # 防止死循环

def evaluate():
    # 1. 初始化配置 
    # 使用 ENV_ID 初始化，确保 config.py 能正确识别 "GoalReach" 并加载 93 维状态
    cfg = Config(algo=ALGO_NAME, env_name=ENV_ID)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 创建环境 (渲染可选 "human" 或 None)
    # 评估时通常想看效果，render_mode=None (Gazebo 自带界面)
    env = gym.make(ENV_ID, render_mode=None, max_steps=MAX_STEPS) 
    
    # 3. 初始化 Agent 并加载模型
    agent = KFDQNAgent(cfg)
    
    # --- 路径处理逻辑 (防止 FileNotFoundError) ---
    abs_model_path = MODEL_PATH
    if not os.path.isabs(MODEL_PATH):
        # 尝试相对于当前工作目录
        if not os.path.exists(abs_model_path):
            # 尝试相对于脚本所在目录
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # 假设 MODEL_PATH 是 "src/scripts/..."，回退两级
            abs_model_path = os.path.join(base_dir, "..", "..", MODEL_PATH)

    try:
        agent.load(abs_model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Tried path: {abs_model_path}")
        print("Please check MODEL_PATH in eval_main.py")
        return

    # [关键] 切换到评估模式
    agent.eval_mode()
    print(f"Agent switched to EVAL mode (Epsilon=0, Network Frozen)")

    # 4. 评估循环
    success_count = 0
    total_rewards = []
    
    print(f"{'-'*30}")
    print(f"Start Evaluation: {EVAL_EPISODES} Episodes")
    print(f"Model: {abs_model_path}")
    print(f"{'-'*30}")

    for i in range(1, EVAL_EPISODES + 1):
        state, _ = env.reset()
        ep_reward = 0
        steps = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # 这里的 take_action 已经是贪婪的了
            action_result = agent.take_action(state)
            
            # [关键修正] 解包元组：KFDQNAgent 返回 (action, strategy, q_action)
            # 如果不加这一步，就会报 AssertionError: assert self.action_space.contains(action)
            if isinstance(action_result, tuple):
                action = action_result[0]
            else:
                action = action_result
            
            next_state, reward, done, truncated, info = env.step(action)
            
            state = next_state
            ep_reward += reward
            steps += 1
            
            if steps >= MAX_STEPS:
                truncated = True

        # 统计
        is_success = info.get('is_success', False)
        if is_success:
            success_count += 1
        total_rewards.append(ep_reward)
        
        print(f"Eval Ep {i}/{EVAL_EPISODES} | Reward: {ep_reward:.2f} | Steps: {steps} | Success: {is_success}")

    # 5. 输出最终结果
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / EVAL_EPISODES
    print(f"{'='*30}")
    print(f"Evaluation Done.")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate:   {success_rate*100:.1f}%")
    print(f"{'='*30}")
    
    env.close()

if __name__ == "__main__":
    evaluate()