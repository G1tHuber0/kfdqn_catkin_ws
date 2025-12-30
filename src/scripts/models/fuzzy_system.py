# models/fuzzy_system.py

import math
import torch
import torch.nn as nn
import itertools
import random
class ROSMobileFuzzyConfig:
    """
    模糊逻辑控制器超参数配置类
    定义了隶属度函数的中心、宽度(Sigma)以及动作推荐的逻辑权重常数
    """
    # 状态量归一化映射: theta_norm 在 [-1,1], lidar_norm 在 [0,1]
    ANTECEDENT_CENTERS = [
        [math.pi, 0.0, -math.pi],   # 维度0 (Theta): 对应 左(Left), 前(Front), 右(Right)
        [0.5, 1.5],      # 维度1 (Lidar): 对应 近(Close, 靠近碰撞线), 远(Far)
    ]
    
    # 高斯隶属度函数的标准差(Sigma)，通过公式 sigma = width/4 换算而来
    ANTECEDENT_SIGMAS = [
        [2, 0.5, 2], # 对应 theta 的 w=[1, 0.75, 1]
        [0.3, 0.3],      # 对应 lidar 的 w=[0.5, 2.75]
    ]

    # 预定义的规则权重极值：支持(Support)或反对(Oppose)
    ACTION_SUPPORT = 1.0
    ACTION_OPPOSE  = -1.0

    # 输入物理限制阈值，用于 preprocess 中的 clamp 操作
    THETA_LIMIT = math.pi
    LIDAR_LIMIT = 3.5


class FuzzySystem(nn.Module):
    """
    KFDQN (Knowledge-augmented Fuzzy Deep Q-Network) 模糊推理层
    功能：模拟模糊推理系统，将传感器状态映射为动作的先验推荐分数
    """
    def __init__(self, device, env_name: str):
        super(FuzzySystem, self).__init__()
        self.device = device
        self.env_name = env_name
        # 识别当前 ROS 任务环境类型
        self.is_goalreach_ros = "GoalReach" in env_name
        self.is_obstacle_avoid_ros = "ObstacleAvoid" in env_name

        # 初始化输入维度与规则总数
        if self.is_goalreach_ros:     # 目标趋近环境
            self.cfg_cls = ROSMobileFuzzyConfig
            self.num_inputs = 1       # 仅角度输入
            self.num_rules = 3        # 3条单变量规则
            self.action_dim = 3
        else:
            # 避障环境
            self.cfg_cls = ROSMobileFuzzyConfig
            self.num_inputs = 2       # 角度 + 距离
            self.num_rules = 6        # 3(角度)*2(距离)=6条组合规则
            self.action_dim = 3
        
        # 1. 初始化高斯隶属度参数为 nn.Parameter (使其具备可学习能力)
        self._init_fuzzy_sets()

        # 2. 定义预处理缩放系数
        if self.is_goalreach_ros:
            self.scales = torch.tensor([math.pi], device=device)
        elif self.is_obstacle_avoid_ros:
            self.scales = torch.tensor([math.pi, 3.5], device=device)
        else:
            pass

        # 3. 初始化规则权重矩阵 [规则数, 动作维度]
        self.rule_weights = nn.Parameter(torch.zeros(self.num_rules, self.action_dim).to(device))
        self._build_rule_base()

    def _init_fuzzy_sets(self):
        """ 将配置中的中心和标准差注册为可学习的模型参数 """
        if self.is_goalreach_ros or self.is_obstacle_avoid_ros:
            self.theta_centers = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_CENTERS[0], device=self.device))
            self.theta_sigmas = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_SIGMAS[0], device=self.device))
            self.lidar_centers = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_CENTERS[1], device=self.device))
            self.lidar_sigmas = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_SIGMAS[1], device=self.device))
            # 以下为针对其他环境预留的占位符
            self.centers = None 
            self.sigmas = None
            self.pos_centers = None
            self.pos_sigmas = None
            self.vel_centers = None
            self.vel_sigmas = None

    def preprocess(self, state):
        """ 数据预处理：执行缩放和截断(Clamp)，防止输入越界导致隶属度消失 """
        # 1. 缩放
        scaled_state = state * self.scales
        
        # 2. 截断
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
        """ 专家规则初始化：将人类先验知识注入 rule_weights 矩阵 """
        if self.is_goalreach_ros:
            SUPPORT = self.cfg_cls.ACTION_SUPPORT
            OPPOSE = self.cfg_cls.ACTION_OPPOSE
            nn.init.constant_(self.rule_weights, OPPOSE)
            with torch.no_grad():
                # 根据角度方向推荐动作 (目标在左推荐左转，以此类推)
                self.rule_weights[2, 1] = SUPPORT # Rule 0: target left
                self.rule_weights[1, 2] = SUPPORT # Rule 1: target front
                self.rule_weights[0, 0] = SUPPORT # Rule 2: target right
            return

        if self.is_obstacle_avoid_ros:
            SUPPORT = self.cfg_cls.ACTION_SUPPORT
            OPPOSE = self.cfg_cls.ACTION_OPPOSE
            nn.init.constant_(self.rule_weights, OPPOSE)
            with torch.no_grad():
                # --- 语义知识 1: 障碍物近 且 目标在左 -> 左转 (Action 0) ---
                # 逻辑: theta_i=0 (左), lidar_i=0 (近) -> idx = 0*2+0 = 0
                self.rule_weights[0, 0] = SUPPORT
                
                # --- 语义知识 3: 障碍物远 且 目标在左 -> 左转 (Action 0) ---
                # 逻辑: theta_i=0 (左), lidar_i=1 (远) -> idx = 0*2+1 = 1
                self.rule_weights[1, 0] = SUPPORT

                # --- 语义知识 2: 障碍物近 且 目标在前 -> 左转 (a0) 或 右转 (a1) ---
                # 逻辑: theta_i=1 (前), lidar_i=0 (近) -> idx = 1*2+0 = 2
                self.rule_weights[2, 0] = SUPPORT # 支持左转避障
                self.rule_weights[2, 1] = SUPPORT 
                self.rule_weights[2, 2] = OPPOSE  # 明确反对直行 (防止碰撞)

                # --- 语义知识 5: 障碍物远 且 目标在前 -> 直行 (Action 2) ---
                # 逻辑: theta_i=1 (前), lidar_i=1 (远) -> idx = 1*2+1 = 3
                self.rule_weights[3, 2] = SUPPORT

                # --- 语义知识 0: 障碍物近 且 目标在右 -> 右转 (Action 1) ---
                # 逻辑: theta_i=2 (右), lidar_i=0 (近) -> idx = 2*2+0 = 4
                self.rule_weights[4, 1] = SUPPORT

                # --- 语义知识 4: 障碍物远 且 目标在右 -> 右转 (Action 1) ---
                # 逻辑: theta_i=2 (右), lidar_i=1 (远) -> idx = 2*2+1 = 5
                self.rule_weights[5, 1] = SUPPORT
            return

    def gaussian(self, x, mu, sigma):
        """ 计算高斯隶属度函数 (模糊化步骤) """
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def forward(self, state):
        """
        推理前向传播流水线：
        特征提取 -> 模糊化 -> 规则激活 -> 解模糊
        """
        # 0. Batch 适配
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch_size = state.shape[0] 

        # --- 第一步：特征映射与提取 ---
        if self.is_goalreach_ros or self.is_obstacle_avoid_ros:
            # 提取目标相对角度 (假设在状态向量索引 90)
            theta_d = state[..., 90] 
            
            if self.is_obstacle_avoid_ros:
                # # 核心修正：从原始 state (0-89位是雷达) 提取数据，而不是从 theta_d 提取
                # # 1. 右前方: 270° 到 360° (对应索引 67 到 89)
                # lidar_right_front = state[..., 67:90] 
                # # 2. 左前方: 0° 到 90° (对应索引 0 到 22)
                # lidar_left_front = state[..., 0:22] 

                # # 拼接前方 180 度区域并取最小值
                # lidar_180 = torch.cat([lidar_right_front, lidar_left_front], dim=-1)
                min_lidar =  state[..., 0:90].min(dim=-1).values

                # 核心修正：将 角度(theta_d) 和 最小值(min_lidar) 堆叠，形成 [Batch, 2] 的特征
                feats = torch.stack([theta_d, min_lidar], dim=-1)
            else:
                # 仅目标趋近任务
                feats = theta_d.unsqueeze(-1)
            
            x_in = self.preprocess(feats)
        else:
            x_in = self.preprocess(state)
        
        # 维度变换以便进行模糊计算 [Batch, Num_Inputs, 1]
        x = x_in.unsqueeze(2)

        # --- 第二步：模糊推理核心 ---
        if self.is_goalreach_ros or self.is_obstacle_avoid_ros:
            # 输入 0: 角度
            theta = x[:, 0, :] # [B, 1]
            mu_theta = self.gaussian(theta, self.theta_centers, self.theta_sigmas) # [B, 3]
            
            if self.is_obstacle_avoid_ros:
                # 输入 1: 雷达最小值
                lidar = x[:, 1, :] # [B, 1]
                mu_lidar = self.gaussian(lidar, self.lidar_centers, self.lidar_sigmas) # [B, 2]
                
                # 计算规则激活强度 (AND 逻辑)
                # [B, 3, 1] bmm [B, 1, 2] -> [B, 3, 2]
                firing = torch.bmm(mu_theta.unsqueeze(2), mu_lidar.unsqueeze(1))
                firing = firing.view(batch_size, -1) # 展平为 [B, 6] 条规则强度
            else:
                firing = mu_theta # 仅单变量 [B, 3]

        # --- 第三步：归一化 ---
        norm = firing / (torch.sum(firing, dim=1, keepdim=True) + 1e-6)
        
        # --- 第四步：解模糊 (加权求和) ---
        output = torch.matmul(norm, self.rule_weights)
        
        return output