import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

from .utils import OrnsteinUhlenbeckNoise  # [修改] 导入共享 OU Noise

# 设置设备 (优先使用 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 神经网络定义 (保持不变)
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

        # [硬件限制] Turtlebot3 Burger 物理极限
        self.max_linear_vel = 0.22  
        self.max_angular_vel = 1

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))

        # 线速度: Sigmoid(0~1) * 0.22 -> [0, 0.22]
        linear = torch.sigmoid(self.linear_head(x)) * self.max_linear_vel
        
        # 角速度: Tanh(-1~1) * 1 -> [-1, 1]
        angular = torch.tanh(self.angular_head(x)) * self.max_angular_vel
        
        return torch.cat([linear, angular], dim=1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 双流输入: State(94->128), Action(2->128)
        self.l1_s = nn.Linear(state_dim, 128)
        self.l1_a = nn.Linear(action_dim, 128)

        # 合并后: 256 -> 256 -> 256 -> 128 -> 64 -> 1
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, 64)
        self.l6 = nn.Linear(64, 1)

    def forward(self, state, action):
        s = F.relu(self.l1_s(state))
        a = F.relu(self.l1_a(action))
        
        x = torch.cat([s, a], dim=1) # Concat
        
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return self.l6(x)

# ==========================================
# 2. 辅助组件 (Buffer & Noise)
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
# 3. DDPG Agent (核心控制类)
# ==========================================

class DDPGAgent:
    def __init__(self, state_dim, action_dim=2):
        # --- 论文 Table 4 超参数 ---
        self.lr_a = 0.0001
        self.lr_c = 0.001
        self.gamma = 0.99
        self.tau = 0.05
        self.batch_size = 256
        self.memory_size = 100000
        self.min_buffer_size = 1500  # 核心：存够数据前不训练

        # 保存物理限制，用于select_action的最终截断
        self.max_linear_vel = 0.22
        self.max_angular_vel = 1.0

        # 初始化组件
        self.memory = ReplayBuffer(self.memory_size)

        # 初始化网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_a)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_c)
        
        self.mse_loss = nn.MSELoss()

        # [修改] 使用 OU 噪声 (Paper Eq.14)
        self.noise = OrnsteinUhlenbeckNoise(action_dim, sigma=[0.05, 0.2], dt=1.0)

    def select_action(self, state, noise=True):
        """
        输入: state (numpy array, shape=(94,))
        输出: action (numpy array, shape=(2,)) -> [v, w]
        """
        # 1. 转换为 Tensor 并传入 GPU
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # 2. 网络推理 (No Grad)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        self.actor.train()

        # 3. 添加噪声 (仅在训练阶段)
        if noise:
            # [修改] 使用非对称高斯噪声
            final_noise = self.noise.sample()
            action[0] += final_noise[0]
            action[1] += final_noise[1]
            
        # 4. 物理截断 (确保指令安全)
        # 即使网络已经有Sigmoid/Tanh限制，加噪后可能会越界，必须Clip
        action[0] = np.clip(action[0], 0.0, self.max_linear_vel)
        action[1] = np.clip(action[1], -self.max_angular_vel, self.max_angular_vel)
        
        return action

    def update(self):
        # 如果经验池数据不足，跳过训练
        if len(self.memory) < self.min_buffer_size:
            return

        # 1. 采样
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        # 2. 更新 Critic
        with torch.no_grad():
            # 计算 Target Action
            next_action = self.actor_target(next_state)
            # 计算 Target Q
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q = self.critic(state, action)
        critic_loss = self.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 3. 更新 Actor
        # 目标：最大化 Critic 对当前策略动作的评分 (即最小化 -Q)
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 4. 软更新 Target Networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=device))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())