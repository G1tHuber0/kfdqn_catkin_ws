#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import gymnasium as gym

# Local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import envs_ros.ros_gazebo_mobile_robot_env  # register envs
from config import Config
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from agents.kfdqn_agent import KFDQNAgent

# =============================
# Config
# =============================
ALGO_NAME = "KFDQN"
ENV_ID = "ObstacleAvoidEval-v0"
MODEL_PATH = "src/scripts/outputs/KFDQN_ObstacleAvoidROS-v0_YYYYMMDD_HHMMSS/models/KFDQN_YYYYMMDD_HHMMSS_final.pth"

EVAL_EPISODES = 100
MAX_STEPS = 1000


def _resolve_model_path(model_path: str) -> str:
    if os.path.isabs(model_path):
        return model_path
    if os.path.exists(model_path):
        return model_path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, "..", "..", model_path))


def _build_agent(cfg: Config):
    if ALGO_NAME == "KFDQN":
        return KFDQNAgent(cfg)
    if ALGO_NAME == "DQN":
        return DQNAgent(cfg)
    if ALGO_NAME == "Double":
        return DoubleDQNAgent(cfg)
    if ALGO_NAME == "Dueling":
        return DuelingDQNAgent(cfg)
    raise ValueError(f"Unsupported ALGO_NAME: {ALGO_NAME}")


def _select_action(agent, state):
    state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.q_net(state_tensor)
        return int(q_values.argmax(dim=1).item())


def evaluate():
    cfg = Config(algo=ALGO_NAME, env_name=ENV_ID)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.h1 is None:
        cfg.h1 = 0.0
    if cfg.h2 is None:
        cfg.h2 = 1.0

    env = gym.make(ENV_ID, render_mode=None, max_steps=MAX_STEPS)

    agent = _build_agent(cfg)
    abs_model_path = _resolve_model_path(MODEL_PATH)
    agent.load(abs_model_path)

    if hasattr(agent, "eval_mode"):
        agent.eval_mode()
    else:
        agent.q_net.eval()

    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0

    success_count = 0
    collision_count = 0
    returns = []

    print("-" * 40)
    print(f"Start Evaluation: {EVAL_EPISODES} Episodes")
    print(f"Env:   {ENV_ID}")
    print(f"Model: {abs_model_path}")
    print("-" * 40)

    for ep in range(1, EVAL_EPISODES + 1):
        state, _ = env.reset()
        done = False
        truncated = False
        ep_return = 0.0
        info = {}
        steps = 0

        while not (done or truncated):
            action = _select_action(agent, state)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            ep_return += float(reward)
            steps += 1
            if steps >= MAX_STEPS:
                truncated = True

        is_success = bool(info.get("is_success", False))
        is_collision = bool(info.get("is_collision", False))
        if is_success:
            success_count += 1
        if is_collision:
            collision_count += 1
        returns.append(ep_return)

        print(
            f"Eval Ep {ep}/{EVAL_EPISODES} | Return: {ep_return:.2f} | "
            f"Steps: {steps} | Success: {is_success} | Collision: {is_collision}"
        )

    avg_return = float(np.mean(returns)) if returns else 0.0
    success_rate = success_count / max(EVAL_EPISODES, 1)
    collision_rate = collision_count / max(EVAL_EPISODES, 1)

    print("=" * 40)
    print("Evaluation Done.")
    print(f"Average Return:  {avg_return:.2f}")
    print(f"Success Rate:    {success_rate * 100:.1f}%")
    print(f"Collision Rate:  {collision_rate * 100:.1f}%")
    print("=" * 40)

    env.close()


if __name__ == "__main__":
    evaluate()
