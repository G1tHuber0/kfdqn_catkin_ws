import math
import torch
import torch.nn as nn

class ROSMobileFuzzyConfig:
    """
    模糊逻辑控制器超参数配置类
    定义了隶属度函数的中心、宽度(Sigma)以及动作推荐的逻辑权重常数
    """
    # 状态量归一化映射: theta_norm 在 [-1,1], lidar_norm 在 [0,1]
    ANTECEDENT_CENTERS = [
        [math.pi/6, 0.0, -math.pi/6],   # Dim 0 (Theta): Left, Front, Right
        [0.0, 2],      # Dim 1 (Lidar Front): Close, Far
        [0.0, 2],      # Dim 2 (Lidar Side): Close, Far
    ]
    
    # 高斯隶属度函数的标准差(Sigma)
    ANTECEDENT_SIGMAS = [
        [0.3, 0.2, 0.3], # Theta
        [0.35,0.8],      # Lidar Front
        [0.35,0.8],      # Lidar Side
    ]

    # 预定义的规则权重极值：支持(Support)或反对(Oppose)
    ACTION_SUPPORT = 1.0
    ACTION_OPPOSE  = -1.0

    # 输入物理限制阈值，用于 preprocess 中的 clamp 操作
    THETA_LIMIT = math.pi
    LIDAR_LIMIT = 3.5
    STOP_DISTANCE = 0.3  # meters (paper Table 2/3 precise knowledge)
    FRONT_SECTOR_HALF_WIDTH = 23  # 90-dim lidar, 4 deg/pt: ~92 degrees


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
        # Env1: goal-reaching (no obstacles); Env2/3/4: obstacle scenarios
        self.is_goalreach_ros = ("GoalReach" in env_name) or (env_name in {"Env1"})
        self.is_obstacle_avoid_ros = ("ObstacleAvoid" in env_name) or (env_name in {"Env2", "Env3", "Env4"})

        # 初始化输入维度与规则总数
        if self.is_goalreach_ros:     # 目标趋近环境
            self.cfg_cls = ROSMobileFuzzyConfig
            self.num_inputs = 1       # 仅角度输入
            self.num_rules = 3        # 3条单变量规则
            self.action_dim = 3
        else:
            # 避障环境
            self.cfg_cls = ROSMobileFuzzyConfig
            self.num_inputs = 4       # Theta, Front, Left, Right
            self.num_rules = 8        # 6 (Front/Theta) + 2 (Side)
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

    def _front_min_lidar_norm(self, state: torch.Tensor) -> torch.Tensor | None:
        """
        从 90 维 LiDAR 归一化输入中提取前方扇区的最小距离 (归一化标度)。
        约定：state[..., 0:90] 为 lidar_norm ∈ [0,1]，且前方方向在 0° 附近，需做首尾拼接。
        """
        if state.shape[-1] < 90:
            return None
        lidar = state[..., 0:90]
        w = int(self.cfg_cls.FRONT_SECTOR_HALF_WIDTH)
        front = torch.cat([lidar[..., :w], lidar[..., -w:]], dim=-1)
        return front.min(dim=-1).values

    def _side_min_lidar_norm(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Extract min distance for Left and Right sectors.
        Assumption: 90 pts, 0=Front, increases CCW.
        Front Sector w=23 => [0:23] U [67:90].
        Left Sector: [23:45] (approx 90 deg ~ 180 deg)
        Right Sector: [45:67] (approx 180 deg ~ 270 deg / -90 deg)
        Normalized inputs.
        """
        if state.shape[-1] < 90:
            return None, None
        lidar = state[..., 0:90]
        w = int(self.cfg_cls.FRONT_SECTOR_HALF_WIDTH)
        # Left: [w, 45]
        left_sector = lidar[..., w:45]
        # Right: [45, 90-w]
        right_sector = lidar[..., 45:90-w]
        
        min_left = left_sector.min(dim=-1).values
        min_right = right_sector.min(dim=-1).values
        return min_left, min_right

    def _init_fuzzy_sets(self):
        """ 将配置中的中心和标准差注册为可学习的模型参数 """
        if self.is_goalreach_ros or self.is_obstacle_avoid_ros:
            self.theta_centers = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_CENTERS[0], device=self.device))
            self.theta_sigmas = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_SIGMAS[0], device=self.device))
            self.lidar_centers = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_CENTERS[1], device=self.device))
            self.lidar_sigmas = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_SIGMAS[1], device=self.device))
            self.lidar_side_centers = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_CENTERS[2], device=self.device))
            self.lidar_side_sigmas = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_SIGMAS[2], device=self.device))
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
            # Input 2 & 3: Left/Right Lidar
            processed[:, 2] = torch.clamp(processed[:, 2], 0.0, self.cfg_cls.LIDAR_LIMIT)
            processed[:, 3] = torch.clamp(processed[:, 3], 0.0, self.cfg_cls.LIDAR_LIMIT)
        else:
            pass
        
        return processed

    def _build_rule_base(self):
        """ 专家规则初始化：将人类先验知识注入 rule_weights 矩阵 """
        # 约定输出维度为 3：
        # - index 0: turn right
        # - index 1: turn left
        # - index 2: go forward
        if self.is_goalreach_ros:
            SUPPORT = self.cfg_cls.ACTION_SUPPORT
            OPPOSE = self.cfg_cls.ACTION_OPPOSE
            nn.init.constant_(self.rule_weights, OPPOSE)
            with torch.no_grad():
                # 根据角度方向推荐动作 (目标在左推荐左转，以此类推)
                # theta fuzzy sets: [Left, Front, Right] -> indices [0,1,2]
                self.rule_weights[0, 1] = SUPPORT # target left  -> turn left
                self.rule_weights[1, 2] = SUPPORT # target front -> go forward
                self.rule_weights[2, 0] = SUPPORT # target right -> turn right
            return

        if self.is_obstacle_avoid_ros:
            SUPPORT = self.cfg_cls.ACTION_SUPPORT
            OPPOSE = self.cfg_cls.ACTION_OPPOSE
            nn.init.constant_(self.rule_weights, OPPOSE)
            with torch.no_grad():
                # idx = theta_i*2 + lidar_i, theta_i: [Left,Front,Right]=[0,1,2], lidar_i: [Close,Far]=[0,1]

                # target left -> turn left (Table 3: Num 0 + hybrid Num 6)
                self.rule_weights[0, 1] = SUPPORT  # left + close
                self.rule_weights[1, 1] = SUPPORT  # left + far
                
                # obstacle close + target front -> turn (avoid) and oppose forward (Table 3: relates to precise stop / hybrid)
                self.rule_weights[2, 0] = SUPPORT  # support right
                self.rule_weights[2, 1] = SUPPORT  # support left
                self.rule_weights[2, 2] = OPPOSE   # oppose forward

                # obstacle far + target front -> go forward (Table 2/3: hybrid move forward)
                self.rule_weights[3, 2] = SUPPORT

                # target right -> turn right (Table 3: Num 1 + hybrid Num 6)
                self.rule_weights[4, 0] = SUPPORT  # right + close
                self.rule_weights[5, 0] = SUPPORT  # right + far

                # --- New Side Rules ---
                # Rule 6: Left Close -> Turn Right
                self.rule_weights[6, 0] = SUPPORT
                # Rule 7: Right Close -> Turn Left
                self.rule_weights[7, 1] = SUPPORT
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
                min_lidar = self._front_min_lidar_norm(state)
                if min_lidar is None:
                    min_lidar = state[..., 0:90].min(dim=-1).values
                
                # Extract Side Lidar
                min_left, min_right = self._side_min_lidar_norm(state)
                # Fallback if None (shouldn't happen given shape check above)
                if min_left is None: min_left = torch.zeros_like(theta_d)
                if min_right is None: min_right = torch.zeros_like(theta_d)

                # 核心修正：将 角度, 前方, 左侧, 右侧 堆叠 [Batch, 4]
                feats = torch.stack([theta_d, min_lidar, min_left, min_right], dim=-1)
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
                # 输入 1: 雷达最小值 (Front)
                lidar_front = x[:, 1, :] # [B, 1]
                mu_lidar_front = self.gaussian(lidar_front, self.lidar_centers, self.lidar_sigmas) # [B, 2] (Close, Far)
                
                # 输入 2: 左侧雷达
                lidar_left = x[:, 2, :] # [B, 1]
                mu_lidar_left = self.gaussian(lidar_left, self.lidar_side_centers, self.lidar_side_sigmas) # [B, 2]

                # 输入 3: 右侧雷达
                lidar_right = x[:, 3, :] # [B, 1]
                mu_lidar_right = self.gaussian(lidar_right, self.lidar_side_centers, self.lidar_side_sigmas) # [B, 2]

                # 1. 前 6 条规则: Theta x Front (AND logic)
                # [B, 3, 1] bmm [B, 1, 2] -> [B, 3, 2] -> view [B, 6]
                firing_base = torch.bmm(mu_theta.unsqueeze(2), mu_lidar_front.unsqueeze(1))
                firing_base = firing_base.view(batch_size, -1) 
                
                # 2. 第 7 条规则: Left Close -> (Turn Right)
                # Index 0 is Close
                firing_rule_6 = mu_lidar_left[:, 0:1] 

                # 3. 第 8 条规则: Right Close -> (Turn Left)
                firing_rule_7 = mu_lidar_right[:, 0:1]

                # 合并所有规则强度 [B, 8]
                firing = torch.cat([firing_base, firing_rule_6, firing_rule_7], dim=1)
            else:
                firing = mu_theta # 仅单变量 [B, 3]

        # --- 第三步：归一化 ---
        norm = firing / (torch.sum(firing, dim=1, keepdim=True) + 1e-6)
        
        # --- 第四步：解模糊 (加权求和) ---
        output = torch.matmul(norm, self.rule_weights)

        # --- Precise knowledge: if obstacle < 0.3m in forward direction, stop (override) ---
        min_front_norm = self._front_min_lidar_norm(state)
        if min_front_norm is not None:
            min_front_m = min_front_norm * float(self.cfg_cls.LIDAR_LIMIT)
            stop_mask = min_front_m < float(self.cfg_cls.STOP_DISTANCE)
            if stop_mask.any():
                output = torch.where(
                    stop_mask.unsqueeze(-1),
                    torch.full_like(output, float(self.cfg_cls.ACTION_OPPOSE)),
                    output,
                )

        return output
