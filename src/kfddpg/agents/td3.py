import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from .utils import OrnsteinUhlenbeckNoise  # [修改]共享工具

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 经验回放池 (Replay Buffer) - [保持不变]
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=256):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.FloatTensor(np.array(action)).to(device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)


# ==========================================
# 2. Actor 网络 - [保持不变]
# ==========================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        # 结构: 94 -> 128 -> 256 -> 256 -> 128 -> 64
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, 64)

        # 输出层：双头
        self.linear_head = nn.Linear(64, 1)   # 线速度
        self.angular_head = nn.Linear(64, 1)  # 角速度

        # [物理/算法限制]
        self.max_linear_vel = 0.22 
        self.max_angular_vel = 1.0 

        # 权重初始化
        nn.init.uniform_(self.linear_head.weight, -0.003, 0.003)
        nn.init.uniform_(self.linear_head.bias, -0.003, 0.003)
        nn.init.uniform_(self.angular_head.weight, -0.003, 0.003)
        nn.init.uniform_(self.angular_head.bias, -0.003, 0.003)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))

        linear = torch.sigmoid(self.linear_head(x)) * self.max_linear_vel
        angular = torch.tanh(self.angular_head(x)) * self.max_angular_vel
        
        return torch.cat([linear, angular], dim=1)


# ==========================================
# 3. Twin Critic 网络 - [修改为双 Q 结构]
# ==========================================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # --- Q1 网络结构 ---
        self.l1_s_1 = nn.Linear(state_dim, 128)
        self.l1_a_1 = nn.Linear(action_dim, 128)
        self.l2_1 = nn.Linear(256, 256)
        self.l3_1 = nn.Linear(256, 256)
        self.l4_1 = nn.Linear(256, 128)
        self.l5_1 = nn.Linear(128, 64)
        self.l6_1 = nn.Linear(64, 1)

        # --- Q2 网络结构 (独立参数) ---
        self.l1_s_2 = nn.Linear(state_dim, 128)
        self.l1_a_2 = nn.Linear(action_dim, 128)
        self.l2_2 = nn.Linear(256, 256)
        self.l3_2 = nn.Linear(256, 256)
        self.l4_2 = nn.Linear(256, 128)
        self.l5_2 = nn.Linear(128, 64)
        self.l6_2 = nn.Linear(64, 1)

    def forward(self, state, action):
        """同时计算 Q1 和 Q2 (用于 Critic 更新)"""
        # Q1
        s1 = F.relu(self.l1_s_1(state))
        a1 = F.relu(self.l1_a_1(action))
        x1 = torch.cat([s1, a1], dim=1)
        x1 = F.relu(self.l2_1(x1))
        x1 = F.relu(self.l3_1(x1))
        x1 = F.relu(self.l4_1(x1))
        x1 = F.relu(self.l5_1(x1))
        q1 = self.l6_1(x1)

        # Q2
        s2 = F.relu(self.l1_s_2(state))
        a2 = F.relu(self.l1_a_2(action))
        x2 = torch.cat([s2, a2], dim=1)
        x2 = F.relu(self.l2_2(x2))
        x2 = F.relu(self.l3_2(x2))
        x2 = F.relu(self.l4_2(x2))
        x2 = F.relu(self.l5_2(x2))
        q2 = self.l6_2(x2)

        return q1, q2

    def Q1(self, state, action):
        """只计算 Q1 (用于 Actor 更新)"""
        s1 = F.relu(self.l1_s_1(state))
        a1 = F.relu(self.l1_a_1(action))
        x1 = torch.cat([s1, a1], dim=1)
        x1 = F.relu(self.l2_1(x1))
        x1 = F.relu(self.l3_1(x1))
        x1 = F.relu(self.l4_1(x1))
        x1 = F.relu(self.l5_1(x1))
        q1 = self.l6_1(x1)
        return q1


# ==========================================
# 4. TD3 Agent - [核心实现]
# ==========================================
class TD3Agent:
    def __init__(self, state_dim, action_dim=2):
        # 基础参数 (保持与 DDPG 一致)
        self.gamma = 0.99
        self.tau = 0.005       
        self.lr_a = 0.0001    # Actor 学习率
        self.lr_c = 0.001     # Critic 学习率
        self.batch_size = 256
        self.memory_size = 100000
        self.min_buffer_size = 1500

        # TD3 特有参数
        self.policy_noise = 0.2     # 目标动作平滑噪声
        self.noise_clip = 0.5       # 噪声截断范围
        self.policy_freq = 2        # 延迟更新频率

        # 物理限制
        self.max_linear_vel = 0.22
        self.max_angular_vel = 1.0

        # [修改] 使用 OU 噪声 (Paper Eq.14)
        self.noise = OrnsteinUhlenbeckNoise(action_dim, sigma=[0.05, 0.2], dt=1.0)

        # 初始化经验池
        self.memory = ReplayBuffer(self.memory_size)
        self.total_it = 0

        # 初始化网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_a)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def select_action(self, state, noise=True):
        """选择动作，使用高斯白噪声"""
        state_tensor = torch.FloatTensor(state).reshape(1, -1).to(device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        self.actor.train()

        if noise:
            # [修改] 使用非对称高斯噪声
            final_noise = self.noise.sample()
            action[0] += final_noise[0]
            action[1] += final_noise[1]

        # 物理截断
        action[0] = np.clip(action[0], 0.0, self.max_linear_vel)
        action[1] = np.clip(action[1], -self.max_angular_vel, self.max_angular_vel)

        return action

    def update(self):
        # 1. 数据检查
        if len(self.memory) < self.min_buffer_size:
            return None, None
        
        self.total_it += 1

        # 2. 采样
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        with torch.no_grad():
            # ============================================
            # [TD3 Trick 1] Target Policy Smoothing
            # ============================================
            # a. 原始目标动作
            next_action = self.actor_target(next_state)

            # b. 生成噪声
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            # c. 叠加噪声
            next_action = next_action + noise

            # d. [关键] 物理截断 (Clip)
            next_action[:, 0] = next_action[:, 0].clamp(0.0, self.max_linear_vel)
            next_action[:, 1] = next_action[:, 1].clamp(-self.max_angular_vel, self.max_angular_vel)

            # ============================================
            # [TD3 Trick 2] Twin Critics (Min Q)
            # ============================================
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # 3. 更新 Critic (Q1 和 Q2 同时更新)
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_val = None

        # ============================================
        # [TD3 Trick 3] Delayed Policy Updates
        # ============================================
        if self.total_it % self.policy_freq == 0:
            
            # 4. 更新 Actor (只用 Q1)
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 5. 软更新 Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            actor_loss_val = actor_loss.item()
            
        return critic_loss.item(), actor_loss_val

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=device))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=device))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)