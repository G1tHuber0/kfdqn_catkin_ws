#!/usr/bin/env python3
import os
import sys
import time
import datetime
import csv
import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# === 导入项目模块 ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import envs_ros.ros_gazebo_mobile_robot_env  # 注册训练环境
from config import Config
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from agents.reinforce_agent import ReinforceAgent
from agents.ac_agent import ACAgent
from agents.kfdqn_agent import KFDQNAgent
from utils.replay_buffer import ReplayBuffer 

# ==========================================
# 1. 全局配置与参数
# ==========================================
ALGO_NAME = "DQN"
ENV_NAME = "GoalReachTrain-v0" 
RENDER_MODE = None             
MAX_STEPS = 100               # 防止死循环
# 自定义模型保存节点 (总步数)
CHECKPOINT_STEPS = [1000, 2000, 4000, 6000, 8000, 10000, 15000,20000,25000,30000,40000,50000]

# 获取时间戳
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 目录定义
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", f"{ALGO_NAME}_{ENV_NAME}_{TIMESTAMP}")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")

# 创建目录
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# 2. 初始化环境与智能体
# ==========================================
def main():
    cfg = Config(algo=ALGO_NAME, env_name=ENV_NAME) 
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make(ENV_NAME, render_mode=RENDER_MODE, max_steps=MAX_STEPS)
    
    np.random.seed(cfg.seed+1)
    torch.manual_seed(cfg.seed+1)
    
    agent = DQNAgent(cfg)
    if hasattr(agent, 'train_mode'):
        agent.train_mode()
    
    replay_buffer = ReplayBuffer(cfg.buffer_size)
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    csv_path = os.path.join(DATA_DIR, "training_log.csv")
    csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Episode", "Total_Steps", "Reward", "Ep_Steps", "Epsilon", "Avg_Loss", "Success"])

    print(f"{'='*40}")
    print(f"   Start Training: {ALGO_NAME}")
    print(f"   Environment:    {ENV_NAME}")
    print(f"   Output Dir:     {OUTPUT_DIR}")
    print(f"{'='*40}\n")

    # ==========================================
    # 3. 训练主循环
    # ==========================================
    total_steps = 0
    start_time = time.time()
    
    # 进度条按“回合数”显示
    # 使用 bar_format 去掉自动的速率显示，避免跳动
    pbar = tqdm(range(1, cfg.episodes + 1), 
                desc="Training", 
                unit="ep",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]"
               )

    for i_episode in pbar:
        state, _ = env.reset(seed=cfg.seed + i_episode)
        ep_reward = 0
        ep_steps = 0
        ep_losses = []
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.take_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            real_done = done and not truncated
            replay_buffer.add(state, action, reward, next_state, real_done)
            
            state = next_state
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            
            # 按步数更新 Epsilon
            agent.update_epsilon(total_steps)
            
            # --- 模型训练 ---
            if replay_buffer.size() > cfg.minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(cfg.batch_size)
                transition_dict = {
                    'states': b_s, 'actions': b_a, 'next_states': b_ns, 
                    'rewards': b_r, 'dones': b_d
                }
                loss = agent.update(transition_dict)
                ep_losses.append(loss)
                
                if total_steps % 100 == 0:
                    writer.add_scalar('Step/Loss', loss, total_steps)

            # --- 模型保存 (按步数) ---
            if total_steps in CHECKPOINT_STEPS:
                save_name = f"{ALGO_NAME}_{TIMESTAMP}_{total_steps}.pth"
                save_path = os.path.join(MODEL_DIR, save_name)
                agent.save(save_path)
                tqdm.write(f">>> [Checkpoint] Model saved: {save_name} at step {total_steps}")


            if total_steps % 10 == 0: # 降低一点刷新频率，避免闪烁
                elapsed_time = time.time() - start_time
                steps_per_sec = total_steps / (elapsed_time + 1e-9)
                pbar.set_postfix({
                    'T_Steps': total_steps,       
                    'S/s': f"{steps_per_sec:.1f}" 
                })

        # --- 回合结束统计 ---
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        is_success = 1 if info.get('is_success', False) else 0
        
        # TensorBoard
        writer.add_scalar('Episode/01-Reward', ep_reward, i_episode)
        writer.add_scalar('Episode/02-Epsilon', agent.epsilon, i_episode)
        writer.add_scalar('Episode/03-Success', is_success, i_episode)
        writer.add_scalar('Episode/04-Avg_Loss', avg_loss, i_episode)
        writer.add_scalar('Episode/05-Steps', ep_steps, i_episode)

        # CSV
        csv_writer.writerow([i_episode, total_steps, ep_reward, ep_steps, agent.epsilon, avg_loss, is_success])
        csv_file.flush()

        # 终端详细打印 (固定间距对齐)
        if i_episode % 1 == 0:
            log_str = (
                f"Ep {i_episode:>2} || "                  
                f"R: {ep_reward:>7.2f} | "               
                f"Steps: {ep_steps:>4} | "               
                f"Loss: {avg_loss:>6.3f} | "             
                f"Eps: {agent.epsilon:.3f} | "           
                f"Success: {bool(is_success)!s:<5}"         
            )
            tqdm.write(log_str)

    # 4. 结束
    final_save_path = os.path.join(MODEL_DIR, f"{ALGO_NAME}_{TIMESTAMP}_final.pth")
    agent.save(final_save_path)
    print(f"\nTraining Finished. Final model saved to: {final_save_path}")
    
    env.close()
    csv_file.close()
    writer.close()

if __name__ == "__main__":
    main()