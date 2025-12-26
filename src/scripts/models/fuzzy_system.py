# models/fuzzy_system.py

import math
import torch
import torch.nn as nn
import itertools

class ROSMobileFuzzyConfig:
    # theta_norm in [-1,1], lidar_norm in [0,1]
    ANTECEDENT_CENTERS = [
        [-0.7, 0.0, 0.7],   # theta_norm: left, front, right
        [0.06, 0.85],       # lidar_norm: close (~collision), far
    ]
    ANTECEDENT_SIGMAS = [
        [0.4, 0.2, 0.4],
        [0.05, 0.25],
    ]

    ACTION_SUPPORT = 1.0
    ACTION_OPPOSE  = -1.0

    THETA_LIMIT = 1.0
    LIDAR_LIMIT = 1.0



class FuzzySystem(nn.Module):
    """
    KFDQN 模糊逻辑控制器
    核心功能: 将环境状态映射为动作推荐分数 (Logits)
    """
    def __init__(self, device, env_name: str):
        super(FuzzySystem, self).__init__()
        self.device = device
        self.env_name = env_name
        self.is_goalreach_ros = "GoalReach" in env_name
        self.is_obstacle_avoid_ros = "ObstacleAvoidROS" in env_name

        
        if self.is_goalreach_ros:     # Goal Reach ROS 环境
            self.cfg_cls = ROSMobileFuzzyConfig
            self.num_inputs = 1
            self.num_rules = 3
            self.action_dim = 3
        else:
            # Obstacle Avoid ROS 环境
            self.cfg_cls = ROSMobileFuzzyConfig
            self.num_inputs = 2
            self.num_rules = 6
            self.action_dim = 3
        
        
        # 1. 初始化模糊集参数
        self._init_fuzzy_sets()

        # 2. 定义缩放系数 (Preprocess Scaling)
        # 将 Gym 微小的数值放大，使其能落在模糊集的有效区间内

        if self.is_goalreach_ros:
            self.scales = torch.tensor([1], device=device)
        elif self.is_obstacle_avoid_ros:
            self.scales = torch.tensor([1.0, 1.0], device=device)
        else:
            pass
        # 3. 初始化规则库权重
        self.rule_weights = nn.Parameter(torch.zeros(self.num_rules, self.action_dim).to(device))
        self._build_rule_base()

    def _init_fuzzy_sets(self):

        if self.is_goalreach_ros or self.is_obstacle_avoid_ros:
            self.theta_centers = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_CENTERS[0], device=self.device))
            self.theta_sigmas = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_SIGMAS[0], device=self.device))
            self.lidar_centers = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_CENTERS[1], device=self.device))
            self.lidar_sigmas = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_SIGMAS[1], device=self.device))
            self.centers = None 
            self.sigmas = None
            self.pos_centers = None
            self.pos_sigmas = None
            self.vel_centers = None
            self.vel_sigmas = None

    def preprocess(self, state):
        """
        数据预处理流水线: Scaling -> Clamping
        """
        # 1. 缩放
        scaled_state = state * self.scales
        
        # 2. 截断 (避免数值越界导致 Gaussian 输出为 0)
        processed = scaled_state.clone()
        if self.is_goalreach_ros:
            processed[:, 0] = torch.clamp(processed[:, 0], -self.cfg_cls.THETA_LIMIT, self.cfg_cls.THETA_LIMIT)
        elif self.is_obstacle_avoid_ros:
            processed[:, 0] = torch.clamp(processed[:, 0], -self.cfg_cls.THETA_LIMIT, self.cfg_cls.THETA_LIMIT)
            processed[:, 1] = torch.clamp(processed[:, 1], 0.0, self.cfg_cls.LIDAR_LIMIT)
        else:
            pass
        
        return processed

    def _build_rule_base(self):

        if self.is_goalreach_ros:
            SUPPORT = self.cfg_cls.ACTION_SUPPORT
            OPPOSE = self.cfg_cls.ACTION_OPPOSE
            nn.init.constant_(self.rule_weights, OPPOSE)
            with torch.no_grad():
                # Rule 0: target left -> action 0
                self.rule_weights[0, 1] = SUPPORT
                # Rule 1: target front -> action 2
                self.rule_weights[1, 2] = SUPPORT
                # Rule 2: target right -> action 1
                self.rule_weights[2, 0] = SUPPORT
            return

        if self.is_obstacle_avoid_ros:
            SUPPORT = self.cfg_cls.ACTION_SUPPORT
            OPPOSE = self.cfg_cls.ACTION_OPPOSE
            nn.init.constant_(self.rule_weights, OPPOSE)
            with torch.no_grad():
                # rule index: theta_i (0=left,1=front,2=right), lidar_i (0=close,1=far)
                def set_rule(theta_i, lidar_i, action, support=True):
                    rule_idx = theta_i * 2 + lidar_i
                    self.rule_weights[rule_idx, action] = SUPPORT if support else OPPOSE

                # close rules (lidar_i = 0)
                set_rule(2, 0, 0, True)  # close & LEFT  -> left turn (a0)
                set_rule(0, 0, 1, True)  # close & RIGHT -> right turn (a1)

                # close & front -> both turns supported, forward opposed（保持）
                set_rule(1, 0, 0, True)
                set_rule(1, 0, 1, True)
                set_rule(1, 0, 2, False)

                # far rules (lidar_i = 1)
                set_rule(2, 1, 0, True)  # far & LEFT  -> a0
                set_rule(0, 1, 1, True)  # far & RIGHT -> a1
                set_rule(1, 1, 2, True)  # far & FRONT -> a2
            return

        

    def gaussian(self, x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def forward(self, state):
        # 0. 维度与预处理
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch_size = state.shape[0] 
        if self.is_goalreach_ros or self.is_obstacle_avoid_ros:
            theta_d = state[..., 90]
            if self.is_obstacle_avoid_ros:
                min_lidar = state[..., 0:90].min(dim=-1).values
                feats = torch.stack([theta_d, min_lidar], dim=-1)
            else:
                feats = theta_d.unsqueeze(-1)
            x_in = self.preprocess(feats)
        else:
            x_in = self.preprocess(state)
        x = x_in.unsqueeze(2)

        # ==========================================
        # Branch 1: ROS Goal/Obstacle Inference
        # ==========================================
        if self.is_goalreach_ros or self.is_obstacle_avoid_ros:
            theta = x[:, 0, :] # [B, 1]
            mu_theta = self.gaussian(theta, self.theta_centers, self.theta_sigmas) # [B, 3]
            if self.is_obstacle_avoid_ros:
                lidar = x[:, 1, :] # [B, 1]
                mu_lidar = self.gaussian(lidar, self.lidar_centers, self.lidar_sigmas) # [B, 2]
                firing = torch.bmm(mu_theta.unsqueeze(2), mu_lidar.unsqueeze(1))
                firing = firing.view(batch_size, -1) # [B, 6]
            else:
                firing = mu_theta # [B, 3]


        # 3. 归一化 (Normalization)
        norm = firing / (torch.sum(firing, dim=1, keepdim=True) + 1e-6)
        
        # 4. 解模糊 (Defuzzification)
        # MountainCar: [B, 6] x [6, 3] -> [B, 3]
        # CartPole:    [B, 16] x [16, 2] -> [B, 2]
        output = torch.matmul(norm, self.rule_weights)
        
        return output
    

    
