import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models.networks import QNet
from utils.exploration import get_linear_decay_epsilon

class DoubleDQNAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.device = cfg.device
        
        self.q_net = QNet(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.target_q_net = QNet(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.update_steps = 0
        self.epsilon = cfg.epsilon_start
        self.is_training = True

    def train_mode(self):
        """切换到训练模式：启用探索，启用网络训练层"""
        self.is_training = True
        self.q_net.train()

    def eval_mode(self):
        """切换到评估模式：关闭探索，冻结网络"""
        self.is_training = False
        self.q_net.eval()

    def take_action(self, state):
        if self.is_training and np.random.random() <= self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        if not self.is_training:
            with torch.no_grad():
                return self.q_net(state_tensor).argmax().item()
        else:
            return self.q_net(state_tensor).argmax().item()

    def update_epsilon(self, episode_idx):
        self.epsilon = get_linear_decay_epsilon(episode_idx, self.cfg)

    def update(self, transition_dict):
        if not self.is_training:
            return 0.0

        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        # --- Double DQN 核心差异 ---
        with torch.no_grad():
            # 1. 使用【当前网络】确定最大动作
            max_action = self.q_net(next_states).argmax(1).view(-1, 1)
            # 2. 使用【目标网络】计算该动作的价值
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
            q_targets = rewards + self.cfg.gamma * max_next_q_values * (1 - dones)
        # -------------------------

        loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        if self.update_steps % self.cfg.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.update_steps += 1
        
        return loss.item()

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path not found: {path}")
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        print(f"Agent loaded model from {path}")
