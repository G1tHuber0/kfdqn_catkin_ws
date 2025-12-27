import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from models.networks import QNet
from utils.exploration import get_linear_decay_epsilon

class DQNAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.device = cfg.device
        
        # 初始化 Q 网络和目标网络
        self.q_net = QNet(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.target_q_net = QNet(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.update_steps = 0
        self.epsilon = cfg.epsilon_start
        
        # [新增] 内部模式标志
        self.is_training = True 
    
    def train_mode(self):
        """切换到训练模式：启用探索，启用网络训练层"""
        self.is_training = True
        self.q_net.train()
        # 恢复当前的 epsilon (如果是线性衰减，由外部 update_epsilon 控制，这里不强行重置)
        
    def eval_mode(self):
        """切换到评估模式：关闭探索，冻结网络"""
        self.is_training = False
        self.q_net.eval()
        # 评估时不使用 epsilon (完全贪婪)
    
    def take_action(self, state, episode_idx):
        self.update_epsilon(episode_idx)
        # [修改] 评估模式下强制贪婪，训练模式下使用 Epsilon-Greedy
        if self.is_training and np.random.random() <= self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            # 评估时不需要计算梯度，稍微快一点
            with torch.no_grad():
                return self.q_net(state_tensor).argmax().item()

    def update(self, transition_dict,episode_idx):
        # [安全检查] 评估模式下禁止更新
        if not self.is_training:
            return 0.0

        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        q_values = self.q_net(states).gather(1, actions)
        
        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.cfg.gamma * max_next_q_values * (1 - dones)

        loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_steps += 1
        if self.update_steps % self.cfg.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            
        return loss.item()

    def update_epsilon(self, step_idx):
        """
        根据当前总步数更新 epsilon
        :param step_idx: 当前的总训练步数 (total_steps)
        """
        self.epsilon = get_linear_decay_epsilon(step_idx, self.cfg)

    # [新增] 模型保存与加载
    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path not found: {path}")
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        print(f"Agent loaded model from {path}")