#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import gymnasium as gym

# Local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs_ros import env_eval  # register envs
from config import Config
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from agents.kfdqn_agent import KFDQNAgent

# =============================
# Config
# =============================
ALGO_NAME = "DuelingDQN"  # 算法名称: DQN, Double, Dueling, KFDQN
ENV_NAME = "GoalReach-v0"  # 使用评估环境   
MODEL_PATH = "src/scripts/outputs/ENV1/DuelingDQN/models/DuelingDQN_20260113_133546_10000.pth"

EVAL_EPISODES = 100
MAX_STEPS = 1000 # 步数保护

def _resolve_model_path(model_path: str) -> str:
    """解析模型路径，支持绝对路径、相对当前路径、相对脚本上级路径"""
    if os.path.isabs(model_path):
        return model_path
    if os.path.exists(model_path):
        return os.path.abspath(model_path)
    # 尝试相对于脚本所在目录向上回退
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, "..", "..", model_path))


def _build_agent(cfg: Config):
    if ALGO_NAME == "KFDQN":
        return KFDQNAgent(cfg)
    if ALGO_NAME == "DQN":
        return DQNAgent(cfg)
    if ALGO_NAME == "Double":
        return DoubleDQNAgent(cfg)
    if ALGO_NAME == "DuelingDQN":
        return DuelingDQNAgent(cfg)
    raise ValueError(f"Unsupported ALGO_NAME: {ALGO_NAME}")


def _select_action(agent, state):
    """
    动作选择逻辑：
    1. KFDQN 特殊处理：关闭 AF/AQ 模式，解包 tuple 
    2. 普通 DQN 处理：直接 argmax
    """
    state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        if "KFDQN" in ALGO_NAME.upper():
            # 评估时通常关闭 AF/AQ 随机性
            agent.use_af = False
            agent.use_aq = False
            action_result = agent.take_action(state, 0) # 0 代表 epsilon=0
            
            # 解包元组 (action, strategy, q_values)
            if isinstance(action_result, (tuple, list)):
                return int(action_result[0])
            return int(action_result)
        
        # 普通 DQN 算法逻辑
        q_values = agent.q_net(state_tensor)
        return int(q_values.argmax(dim=1).item())


def evaluate():
    # 1. 初始化配置
    cfg = Config(algo=ALGO_NAME, env_name=ENV_NAME)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 兼容性设置
    if hasattr(cfg, 'h1') and cfg.h1 is None: cfg.h1 = 0.0
    if hasattr(cfg, 'h2') and cfg.h2 is None: cfg.h2 = 1.0

    # 2. 创建环境
    env = gym.make(ENV_NAME, render_mode=None)

    # 3. 初始化 Agent 并加载模型
    agent = _build_agent(cfg)
    abs_model_path = _resolve_model_path(MODEL_PATH)
    
    if not os.path.exists(abs_model_path):
        print(f"Error: Model not found at {abs_model_path}")
        return
        
    agent.load(abs_model_path)

    # 切换到评估模式
    if hasattr(agent, "eval_mode"):
        agent.eval_mode()
    elif hasattr(agent, "q_net"):
        agent.q_net.eval()

    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0

    # 4. 评估循环
    success_count = 0
    collision_count = 0
    returns = []

    print("-" * 40)
    print(f"Start Evaluation: {EVAL_EPISODES} Episodes")
    print(f"Env:    {ENV_NAME}")
    print(f"Algo:   {ALGO_NAME}")
    print(f"Model:  {abs_model_path}")
    print("-" * 40)

    for ep in range(1, EVAL_EPISODES + 1):
        # 使用 cfg.seed 或 ep 保证可复现性
        state, _ = env.reset(seed=cfg.seed + ep if cfg.seed else None)
        terminated = False
        truncated = False
        ep_return = 0.0
        steps = 0
        info = {}

        while not (terminated or truncated):
            action = _select_action(agent, state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            state = next_state
            ep_return += float(reward)
            steps += 1
            
            if steps >= MAX_STEPS:
                truncated = True

        # 统计
        is_success = bool(info.get("is_success", False))
        is_collision = bool(info.get("is_collision", False))
        
        if is_success:
            success_count += 1
        if is_collision:
            collision_count += 1
            
        returns.append(ep_return)

        print(
            f"Eval Ep {ep:3d}/{EVAL_EPISODES} | Return: {ep_return:7.2f} | "
            f"Steps: {steps:4d} | Success: {str(is_success):5s} | Collision: {is_collision}"
        )

    # 5. 输出汇总
    avg_return = float(np.mean(returns)) if returns else 0.0
    success_rate = success_count / max(EVAL_EPISODES, 1)
    collision_rate = collision_count / max(EVAL_EPISODES, 1)

    print("=" * 40)
    print("Evaluation Summary")
    print(f"Average Return:  {avg_return:.2f}")
    print(f"Success Rate:    {success_rate * 100:.1f}%")
    print(f"Collision Rate:  {collision_rate * 100:.1f}%")
    print("=" * 40)

    env.close()


if __name__ == "__main__":
    evaluate()