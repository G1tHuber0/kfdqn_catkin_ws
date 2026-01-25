import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .fuzzy_system import FuzzySystem  # [KF-DDPG] 导入模糊系统
from .utils import OrnsteinUhlenbeckNoise  # [修改] 导入共享 OU Noise

# 设置设备 (优先使用 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 神经网络定义 (严格遵循论文 Fig.5)
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
        self.max_angular_vel = 1.0

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))

        # 线速度: Sigmoid(0~1) * 0.22 -> [0, 0.22]
        linear = torch.sigmoid(self.linear_head(x)) * self.max_linear_vel
        
        # 角速度: Tanh(-1~1) * 1.0 -> [-1.0, 1.0]
        # 注意：这里和 TD3/DDPG 保持一致，而非 2.84，因为环境层 action_space 也是 1.0
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

# [已移除] OrnsteinUhlenbeckNoise 类定义，改用 utils.py 导入


# ==========================================
# 3. KF-DDPG Agent (核心算法)
# ==========================================

class KFDDPGAgent:
    """
    Knowledge Guided Deep Deterministic Policy Gradient (KF-DDPG)
    复现论文核心：
    1. 引入 FuzzySystem 作为先验知识 (Knowledge Module)
    2. 在动作选择阶段进行知识引导 (Action Fusion)
    3. 使用 OU noise (paper Eq.19) 探索
    4. 使用 stacked generalization (Eq.17) + 可学习 θ_G (Eq.18)
    5. 使用 supervised-then-reinforced loss (Eq.27)
    """
    def __init__(self, state_dim, action_dim=2, env_name: str = "Env1", *, theta_t: float | None = None):
        # --- 论文 Table 4 超参数 ---
        self.lr_a = 0.0001
        self.lr_c = 0.001
        self.lr_g = 0.001
        self.gamma = 0.99
        self.tau = 0.05
        self.batch_size = 256
        self.memory_size = 100000
        self.min_buffer_size = 1500 

        # 物理限制
        self.max_linear_vel = 0.22
        self.max_angular_vel = 1.0

        # 初始化组件
        self.memory = ReplayBuffer(self.memory_size)
        
        # [KF-DDPG Key 1]: 初始化模糊系统
        self.env_name = str(env_name)
        self.fuzzy_system = FuzzySystem(device, env_name=self.env_name)
        
        # [KF-DDPG Key 2]: 知识引导比率 (eta, for external scheduling / backward compatibility)
        self.knowledge_ratio = 0.6 

        # Table 4: θ_T is fixed (Env1/Env4: 0.5, Env2/Env3: 0)
        if theta_t is None:
            self.theta_T = 0.5 if self.env_name in {"Env1", "Env4", "GoalReach"} else 0.0
        else:
            self.theta_T = float(theta_t)

        # [KGDDPG] stacked generalization parameter θ_G (Eq.17), optimized by Eq.18
        theta_g_init = 1.0 - float(self.knowledge_ratio)  # start with more knowledge guidance
        theta_g_init = float(np.clip(theta_g_init, 1e-4, 1.0 - 1e-4))
        self.theta_g_logit = nn.Parameter(torch.tensor(np.log(theta_g_init / (1.0 - theta_g_init)), device=device))
        self.theta_g_optimizer = optim.Adam([self.theta_g_logit], lr=self.lr_g)
        self._theta_g_override: float | None = None

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

        # [KGDDPG] OU noise (sigma per dimension)
        self.noise = OrnsteinUhlenbeckNoise(action_dim, sigma=[0.05, 0.2], dt=1.0)

    def _theta_g(self) -> torch.Tensor:
        return torch.sigmoid(self.theta_g_logit)

    def _scores_to_action_torch(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Adapter: map fuzzy scores (B,3): [right, left, forward] -> (B,2): [v, w].
        """
        v = torch.clamp((scores[:, 2] + 1.0) / 2.0, 0.0, 1.0) * self.max_linear_vel
        w_score = (scores[:, 1] - scores[:, 0]) / 2.0
        w = torch.clamp(w_score, -1.0, 1.0) * self.max_angular_vel
        return torch.stack([v, w], dim=1)

    def _knowledge_action_torch(self, state_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            scores = self.fuzzy_system(state_tensor)
            return self._scores_to_action_torch(scores)

    def select_action(self, state, noise=True, current_ratio=None):
        """
        动作选择函数 - KF-DDPG 核心部分
        输入: 
            state: 环境状态
            noise: 是否添加随机探索噪声
            current_ratio: 当前的知识融合比例 (如果不传则使用 self.knowledge_ratio)
        """
        # 1. 预处理
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        eta = current_ratio if current_ratio is not None else None

        # 2. 获取 Actor 网络的动作 (Deep Policy)
        self.actor.eval()
        with torch.no_grad():
            actor_action = self.actor(state_tensor).cpu().data.numpy().flatten()
        self.actor.train()

        # 3. [KF-DDPG Core]: 获取 Fuzzy 系统建议动作 (Knowledge Policy)
        with torch.no_grad():
            fuzzy_action = self._knowledge_action_torch(state_tensor).cpu().numpy().flatten()

        # 4. 融合动作 (Stacked Generalization, Eq.17)
        # a_t = θ_G a_μ + (1-θ_G) a_ks
        if eta is None:
            theta_g = float(self._theta_g().detach().cpu().item())
            self._theta_g_override = None
        else:
            theta_g = float(np.clip(1.0 - float(eta), 0.0, 1.0))
            self._theta_g_override = theta_g
        final_action = theta_g * actor_action + (1.0 - theta_g) * fuzzy_action

        # 5. 添加随机噪声 (Exploration Noise)
        # Eq.19: OU noise
        if noise:
            final_action += self.noise.sample()
            
        # 6. 物理截断 (Safety Clip)
        final_action[0] = np.clip(final_action[0], 0.0, self.max_linear_vel)
        final_action[1] = np.clip(final_action[1], -self.max_angular_vel, self.max_angular_vel)
        
        return final_action

    def update(self):
        # 如果经验池数据不足，跳过训练
        if len(self.memory) < self.min_buffer_size:
            return

        # 1. 采样
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        if self._theta_g_override is None:
            theta_g_detached = self._theta_g().detach()
        else:
            theta_g_detached = torch.tensor(float(self._theta_g_override), device=device)

        # 2. 更新 Critic (paper Eq.23: use fused target action a_{G\\bar{θ}})
        with torch.no_grad():
            next_action_mu = self.actor_target(next_state)
            next_action_ks = self._knowledge_action_torch(next_state)
            next_action_fused = theta_g_detached * next_action_mu + (1.0 - theta_g_detached) * next_action_ks
            target_Q = self.critic_target(next_state, next_action_fused)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q = self.critic(state, action)
        critic_loss = self.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 3. 更新 Actor (paper Eq.27: supervised-then-reinforced hybrid loss)
        action_mu = self.actor(state)
        q_mu = self.critic(state, action_mu).mean()
        action_ks = self._knowledge_action_torch(state)
        action_fused_target = theta_g_detached * action_mu.detach() + (1.0 - theta_g_detached) * action_ks
        supervised_loss = self.mse_loss(action_mu, action_fused_target)
        actor_loss = (-theta_g_detached * q_mu) + ((1.0 - theta_g_detached + self.theta_T) * supervised_loss)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 4. 更新 θ_G (paper Eq.18: maximize Q(s, a_G))
        # 若 select_action 外部传入 current_ratio，则认为使用外部日程，不更新 θ_G。
        if self._theta_g_override is None:
            theta_g = self._theta_g()
            with torch.no_grad():
                action_mu_detached = action_mu.detach()
                action_ks_detached = action_ks.detach()
            action_fused = theta_g * action_mu_detached + (1.0 - theta_g) * action_ks_detached
            theta_g_loss = -self.critic(state, action_fused).mean()
            self.theta_g_optimizer.zero_grad()
            theta_g_loss.backward()
            self.theta_g_optimizer.step()

        # 5. 软更新 Target Networks
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
