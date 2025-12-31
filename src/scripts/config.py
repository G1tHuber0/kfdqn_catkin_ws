import torch
import os

class Config:
    def __init__(self, algo='DQN', env_name='GoalReachTrain-v0', **kwargs):
        """
        :param algo: 算法名称
        :param env_name: 环境名称 (支持 GoalReach 或 ObstacleAvoid)
        :param kwargs: 【最高优先级】手动覆盖参数，如 Config(..., lr=0.01)
        """
        # --- 1. 基础属性 ---
        self.algo = algo
        self.env_name = env_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 随机种子处理
        default_seed = 123
        try:
            self.seed = int(os.environ.get("TRAIN_SEED", str(default_seed)))
        except Exception:
            self.seed = default_seed

        # --- 2. 初始化默认参数占位符 (防止 AttributeError) ---
        self._init_placeholders()

        # --- 3. 加载算法参数 (Tier 1: 算法基准) ---
        self._set_algo_params()

        # --- 4. 加载 ROS 环境参数 (Tier 2: 环境覆盖算法) ---
        # 因为是 ROS 专用版，这里会覆盖掉算法中不适合 ROS 的参数(如 LR)
        self._set_ros_env_params()

        # --- 5. 手动参数覆盖 (Tier 3: 最终决定权) ---
        # 允许你在代码运行时动态调整任何参数
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _init_placeholders(self):
        """初始化所有可能用到的属性，避免条件分支导致的未定义错误"""
        # 通用
        self.state_dim = 93
        self.action_dim = 3
        self.hidden_dim = [128, 256, 128, 64]
        self.gamma = 0.99
        self.lr = 1e-4
        self.grad_clip_norm = None
        
        # 探索相关
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.decay_start = 0
        self.decay_steps = 1000

        # KFDQN 特有
        self.h1 = None
        self.h2 = None
        self.m_base = None
        self.use_hybrid_action = False

    def _set_algo_params(self):
        """根据算法设定基础超参"""
        
        # Group A: DQN Family
        if self.algo in ['DQN', 'DoubleDQN', 'DuelingDQN']:
            self.buffer_size = 10000
            self.minimal_size = 1500  # ROS 建议稍微多存一点再开始
            self.batch_size = 256     # ROS 环境通常状态维数大，Batch大一点稳定
            self.target_update = 500
            self.train_freq = 1
            self.gradient_steps = 1
            
            if self.algo == "DuelingDQN":
                self.hidden_dim = [128, 256, 128]

        # Group B: Reinforce
        elif self.algo == 'Reinforce':
            self.buffer_size = None
            self.minimal_size = None
            self.batch_size = None
            self.target_update = None

        # Group C: Actor-Critic
        elif self.algo == 'AC':
            self.buffer_size = None
            self.minimal_size = 0
            self.batch_size = 1
            self.lr_actor = 1e-5
            self.lr_critic = 1e-4

        # Group D: KFDQN
        elif self.algo == 'KFDQN':
            self.use_hybrid_action = True
            self.use_hybrid_learning = True
            
            # 基础训练参数
            self.buffer_size = 10000
            self.minimal_size = 1500
            self.batch_size = 256
            self.target_update = 500
            self.train_freq = 1
            
            # KFDQN 核心参数 (论文/经验值)
            self.h1 = 0.2
            self.h2 = 0.8
            self.ep_r = 25
            self.C_update = 10
            self.m_base = 0.8
            self.m_decay = 0.6
            self.m_tau = 100
            
            self.freeze_fuzzy_premise = True
            self.fuzzy_lr = 0.0001

    def _set_ros_env_params(self):
        """ROS 环境的强约束配置 (覆盖上述参数)"""
        
        # 1. 物理/网络结构强约束
        self.state_dim = 93
        self.action_dim = 3
        # 如果不是 DuelingDQN (Dueling在上面处理了)，保持默认结构
        if self.algo != "DuelingDQN":
            self.hidden_dim = [128, 256, 128, 64]
            
        # 2. 训练超参 (ROS 环境通常需要更小的 LR 和特定的 Gamma)
        self.gamma = 0.99
        self.lr = 1e-4        # ROS 常用 1e-4，比通用 RL 的 1e-3 小
        
        # 3. 探索策略 (Epsilon Greedy)
        # 探索相关
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.decay_start = 0
        if self.algo == 'KFDQN':
            self.epsilon_start = 1.0
            self.epsilon_end = 0.01
            self.decay_start = 0
        # 4. 针对不同 ROS 任务的差异化配置
        if "ObstacleAvoid" in self.env_name:
            # 避障任务通常更难，需要更多轮次和更慢的衰减
            self.episodes = 1000
            self.decay_steps = 2000 
        else:
            # 默认为 GoalReach (寻路任务)
            self.episodes = 500
            self.decay_steps = 1000

    def __repr__(self):
        return f"<Config(ROS) algo={self.algo} env={self.env_name} device={self.device} lr={self.lr}>"