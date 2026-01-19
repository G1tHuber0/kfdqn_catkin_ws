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

# === 导入自定义工具与模块 ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs_ros import env_train  # noqa: F401
from config import Config
from agents.kfdqn_agent import KFDQNAgent
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from utils.replay_buffer import ReplayBuffer
from utils.run_recorder import RunRecorder
from utils.seeding import seed_everything, episode_seed

# ==========================================
# 1. 全局配置与参数 (对齐 Env2 风格)
# ==========================================
ALGO_NAME = os.environ.get("ALGO_NAME", "DubleDQN")
ENV_NAME = "GoalReachTrain-v0"
RENDER_MODE = None

CONTINUE_ON_SUCCESS = False
OUTPUT_ENV_DIR = "ENV1"  # 对应输出的子文件夹

MAX_EPISODES = 500
MAX_TRAIN_STEPS = 99999999
MAX_EPISODE_STEPS = 100

# 统计窗口大小
SUCCESS_WINDOW_SIZE = 100 
COLLISION_WINDOW_SIZE = 100 

CHECKPOINT_STEPS = [1000, 2000, 4000, 6000, 8000, 10000, 15000, 20000, 25000, 30000, 40000, 50000]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main() -> None:
    # 1.1 初始化配置与种子
    cfg = Config(algo=ALGO_NAME, env_name=ENV_NAME)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed_override = os.environ.get("SEED")
    if seed_override is not None:
        cfg.seed = int(seed_override)
    seed_global = cfg.seed

    # 1.2 目录初始化
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{ALGO_NAME}_seed{cfg.seed}_{timestamp}"
    output_dir = os.path.join(BASE_DIR, "outputs", OUTPUT_ENV_DIR, run_name)
    log_dir = os.path.join(output_dir, "logs")
    model_dir = os.path.join(output_dir, "models")
    data_dir = os.path.join(output_dir, "data")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 1.3 环境与种子设置
    env = gym.make(
        ENV_NAME,
        render_mode=RENDER_MODE,
        max_steps=MAX_EPISODE_STEPS,
        continue_on_success=CONTINUE_ON_SUCCESS,
    )
    seed_everything(seed_global, env=env, deterministic_torch=True)

    # 1.4 Agent 初始化
    if ALGO_NAME == "KFDQN":
        agent = KFDQNAgent(cfg)
    elif ALGO_NAME == "DoubleDQN":
        agent = DoubleDQNAgent(cfg)
    elif ALGO_NAME == "DuelingDQN":
        agent = DuelingDQNAgent(cfg)
    else:
        agent = DQNAgent(cfg)
    agent.train_mode()

    replay_buffer = ReplayBuffer(cfg.buffer_size, seed=seed_global)
    writer = SummaryWriter(log_dir=log_dir)

    # 1.5 统计记录器初始化
    success_window = deque(maxlen=SUCCESS_WINDOW_SIZE)
    collision_window = deque(maxlen=COLLISION_WINDOW_SIZE)
    recorder = RunRecorder(data_dir=data_dir, algo_name=ALGO_NAME, env_name=ENV_NAME, timestamp=timestamp)

    # 保存配置到文件
    script_params = {
        "max_episodes": MAX_EPISODES,
        "max_train_steps": MAX_TRAIN_STEPS,
        "success_window_size": SUCCESS_WINDOW_SIZE,
    }
    recorder.save_config(cfg=cfg, agent=agent, env=env, seed_global=seed_global, script_params=script_params, paths={"output_dir": output_dir})

    csv_path = os.path.join(data_dir, "training_log.csv")
    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Episode", "Total_Steps", "Reward", "Ep_Steps", "Epsilon", "Avg_Loss", "Success", "Collision", "Consistency"])

    print(f"{'='*40}")
    print(f"   Start Training: {ALGO_NAME} ({ENV_NAME})")
    print(f"   Output Dir: {output_dir}")
    print(f"{'='*40}\n")

    total_steps = 0
    start_time = time.time()
    pbar = tqdm(range(1, MAX_EPISODES + 1), desc="Training", unit="ep")

    # ==========================================
    # 2. 训练主循环
    # ==========================================
    for i_episode in pbar:
        if total_steps >= MAX_TRAIN_STEPS: break

        # 种子同步：解决随机漂移
        ep_seed = episode_seed(seed_global, i_episode)
        seed_everything(ep_seed, env=env)
        state, _ = env.reset(seed=ep_seed)

        ep_reward, ep_steps = 0.0, 0
        ep_losses, ep_q_losses, ep_fuzzy_losses = [], [], []
        done = False
        
        # KFDQN 特有统计
        kfdqn_stats = {"hybrid_steps": 0, "consistent_steps": 0}

        while not done:
            if total_steps >= MAX_TRAIN_STEPS: break

            # 2.1 动作采样逻辑
            if ALGO_NAME == "KFDQN":
                kfdqn_m, kfdqn_n = agent.update_parameters(i_episode, current_steps=total_steps)
                action_result = agent.take_action(state, i_episode)
            else:
                action_result = agent.take_action(state, total_steps)

            # 解析返回元组 (KFDQN) 或 整数 (DQN)
            q_choice = None
            strategy_tag = "unknown"
            if isinstance(action_result, tuple):
                action = action_result[0]
                if len(action_result) >= 2: strategy_tag = action_result[1]
                if len(action_result) >= 3: q_choice = action_result[2]
            else:
                action = action_result

            # 统计 KFDQN 决策一致性 (Hybrid Action 阶段)
            if ALGO_NAME == "KFDQN" and strategy_tag == 'hya':
                kfdqn_stats["hybrid_steps"] += 1
                if q_choice is not None and action == q_choice:
                    kfdqn_stats["consistent_steps"] += 1

            # 2.2 环境交互
            next_state, reward, terminal, truncated, info = env.step(action)
            done = terminal or truncated
            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            ep_reward += float(reward)
            ep_steps += 1
            total_steps += 1

            # 2.3 网络更新
            if replay_buffer.size() > cfg.minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(cfg.batch_size)
                transition_dict = {"states": b_s, "actions": b_a, "next_states": b_ns, "rewards": b_r, "dones": b_d}
                loss_info = agent.update(transition_dict, episode_idx=i_episode)

                # 处理 Loss 字典 (KFDQN)
                q_loss_val = float(loss_info.get("q_loss", 0.0)) if isinstance(loss_info, dict) else float(loss_info)
                fz_loss_val = float(loss_info.get("fuzzy_loss", 0.0)) if isinstance(loss_info, dict) else 0.0
                curr_loss = q_loss_val + fz_loss_val

                ep_losses.append(curr_loss)
                ep_q_losses.append(q_loss_val)
                ep_fuzzy_losses.append(fz_loss_val)

                if total_steps % 10 == 0:
                    writer.add_scalar("Step/01-Loss", curr_loss, total_steps)

            # 2.4 保存 Checkpoint
            if total_steps in CHECKPOINT_STEPS:
                save_path = os.path.join(model_dir, f"{ALGO_NAME}_{timestamp}_{total_steps}.pth")
                agent.save(save_path)
                tqdm.write(f">>> [Checkpoint] Step {total_steps} saved.")

            if total_steps % 10 == 0:
                pbar.set_postfix({"T_Steps": total_steps, "S/s": f"{total_steps/(time.time()-start_time):.1f}"})

        # ==========================================
        # 3. 回合结束：统计与日志
        # ==========================================
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        is_success = 1 if info.get("is_success", False) else 0
        is_collision = 1 if info.get("is_collision", False) else 0
        
        success_window.append(is_success)
        collision_window.append(is_collision)
        avg_sr = sum(success_window) / SUCCESS_WINDOW_SIZE
        avg_cr = sum(collision_window) / SUCCESS_WINDOW_SIZE
        
        consistency_rate = kfdqn_stats["consistent_steps"] / kfdqn_stats["hybrid_steps"] if kfdqn_stats["hybrid_steps"] > 0 else 0.0

        # Tensorboard 记录
        writer.add_scalar("Episode/01-Reward", ep_reward, i_episode)
        writer.add_scalar(f"Episode/03-SuccessRate_Last{SUCCESS_WINDOW_SIZE}", avg_sr, i_episode)
        writer.add_scalar(f"Episode/07-Collision{COLLISION_WINDOW_SIZE}", avg_cr, i_episode)
        writer.add_scalar("Episode/04-AvgLoss", avg_loss, i_episode)
        
        if ALGO_NAME == "KFDQN":
            writer.add_scalar("KFDQN/01-Consistency", consistency_rate, i_episode)
            writer.add_scalar("KFDQN/02-HybridWeight_m", kfdqn_m, i_episode)

        # CSV & Log
        csv_writer.writerow([i_episode, total_steps, ep_reward, ep_steps, agent.epsilon, avg_loss, is_success, is_collision, consistency_rate])
        csv_file.flush()

        tqdm.write(f"Ep {i_episode:<3} | R: {ep_reward:>6.1f} | SR: {avg_sr:.2f} | Cons: {consistency_rate:.2f} | Success: {bool(is_success)}")

    # 4. 结束与保存
    final_path = os.path.join(model_dir, f"{ALGO_NAME}_{timestamp}_final.pth")
    agent.save(final_path)
    recorder.save_summary(total_steps=total_steps, episodes_completed=i_episode, duration_sec=time.time()-start_time, final_model_path=final_path, metrics={"avg_sr": avg_sr})
    
    env.close()
    csv_file.close()
    writer.close()

if __name__ == "__main__":
    main()