from __future__ import annotations

import math
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
from typing import Optional, Tuple
import time
import rospkg
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray

# --- 检查 Gazebo 服务依赖 ---
try:
    from gazebo_msgs.srv import SetModelState
    from gazebo_msgs.msg import ModelState
    _HAS_MODEL_STATE = True
except Exception:
    _HAS_MODEL_STATE = False

try:
    from gazebo_msgs.srv import DeleteModel, SpawnModel
    _HAS_MODEL_SPAWN = True
except Exception:
    _HAS_MODEL_SPAWN = False

_ROS_INITIALIZED = False

def ensure_ros_init(node_name: str = "gym_mobile_robot_env") -> None:
    """确保 ROS 节点只初始化一次，防止多重初始化报错"""
    global _ROS_INITIALIZED
    if _ROS_INITIALIZED or rospy.core.is_initialized():
        return
    rospy.init_node(node_name, anonymous=True, disable_signals=True)
    _ROS_INITIALIZED = True

def _wrap_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi] 范围"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """从四元数计算偏航角 (Yaw)"""
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)

class ROSGazeboMobileRobotEnv(gym.Env):
    """
    ROS Gazebo 移动机器人强化学习环境
    支持：
    1. 离散动作空间 (左转, 右转, 前进)
    2. 连续状态空间 (90维雷达 + 相对位置信息)
    3. 训练模式 (随机起点/终点/障碍物) 与 评估模式 (固定路径) 切换
    4. Rviz 可视化集成
    """
    metadata = {"render_modes": []}
    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_FORWARD = 2

    def __init__(
        self,
        *,
        # --- ROS 话题配置 ---
        scan_topic: str = "/scan",
        odom_topic: str = "/odom",
        cmd_vel_topic: str = "/cmd_vel",
        reset_world_service: str = "/gazebo/reset_world",
        reset_sim_service: str = "/gazebo/reset_simulation",
        set_model_state_service: str = "/gazebo/set_model_state",
        robot_model_name: str | None = None,
        
        # --- 机器人初始状态 ---
        init_x: float = 0.0,
        init_y: float = 0.0,
        init_yaw: float = 0.0,
        
        # --- 训练参数 ---
        max_steps: int = 100,          # 单回合最大步数
        max_lidar_range: float = 3.5,  # 雷达最大探测距离
        forward_v: float = 0.11,       # 直行线速度
        turn_v: float = 0.11,          # 转向线速度
        turn_omega: float = 1.4,       # 转向角速度
        publish_hz: float = 30.0,      # 控制指令发布频率
        action_duration: float = 0.1,  # 动作持续时间 (秒)
        
        # --- 奖励函数参数 ---
        RTH: float = 0.20,             # 到达目标阈值 (Reach Threshold)
        CTH: float = 0.15,             # 碰撞阈值 (Collision Threshold)
        r_reach: float = 200.0,        # 到达奖励
        r_collision: float = -150.0,   # 碰撞惩罚
        p_r: float = 10,               # 距离引导奖励系数 (Potential Reward)
        r_o: float = -0.1,             # 每步时间惩罚 (Time penalty)
        
        # --- 目标设置 ---
        goal_x: float | None = None,
        goal_y: float | None = None,
        waypoints: list[tuple[float, float]] | None = None, # 评估模式下的路径点列表
        waypoint_rth: float = 0.20,    # 路径点到达判定范围
        random_goal: bool = False,     # (旧参数兼容) 是否启用随机目标
        max_goal_distance: float = 8.0,# 目标最大生成距离
        
        # --- 系统配置 ---
        wait_timeout: float = 1.0,     # 服务等待超时时间
        
        # --- 动态障碍物配置 (Obstacle Mode) ---
        obstacle_mode: bool = False,
        obstacle_models_root: str | None = None,
        train_box_model_dir: str = "train_box",
        train_cyl_model_dir: str = "train_cyl",
        train_tri_model_dir: str = "train_tri",
        arena_half_size: float = 2.5,  # 训练场地半径
        wall_margin: float = 0.2,      # 离墙安全距离
        obstacle_min_dist_to_robot: float = 0.6,
        obstacle_min_dist_to_goal: float = 0.6,
        obstacle_min_dist_between: float = 0.45,
        num_train_obstacles: int = 2,
        debug_obstacles: bool = False,
        obstacle_move_every_steps: int = 15, # 每多少步移动一次障碍物
        obstacle_move_jitter: float = 0.35,  # 障碍物移动幅度
        obstacle_move_max_trials: int = 20,
        
        # --- 可视化 ---
        enable_viz: bool = True,
        viz_frame: str = "odom",
        max_path_len: int = 3000,
        render_mode: str | None = None,
        
        # --- [核心模式开关] ---
        is_training: bool = False, 
    ):
        super().__init__()
        ensure_ros_init()

        # 保存 ROS 配置
        self.scan_topic = scan_topic
        self.odom_topic = odom_topic
        self.cmd_vel_topic = cmd_vel_topic
        self.reset_world_service = reset_world_service
        self.reset_sim_service = reset_sim_service
        self.set_model_state_service = set_model_state_service
        self.robot_model_name = robot_model_name
        
        # 保存参数
        self.init_x = init_x
        self.init_y = init_y
        self.init_yaw = init_yaw
        self.max_steps = max_steps
        self.max_lidar_range = max_lidar_range
        self.forward_v = forward_v
        self.turn_v = turn_v
        self.turn_omega = turn_omega
        self.publish_hz = publish_hz
        self.action_duration = action_duration
        self.RTH = RTH
        self.CTH = CTH
        self.r_reach = r_reach
        self.r_collision = r_collision
        self.p_r = p_r
        self.r_o = r_o
        self.random_goal = random_goal
        self.waypoints = waypoints
        self.waypoint_rth = waypoint_rth
        self.wp_idx = 0
        self.max_goal_distance = max_goal_distance
        self.wait_timeout = wait_timeout
        self.obstacle_mode = obstacle_mode
        
        # 自动查找障碍物模型路径
        if obstacle_models_root is None:
            obstacle_models_root = os.path.join(
                rospkg.RosPack().get_path("kfdqn_gazebo"),
                "models",
            )
        self.obstacle_models_root = obstacle_models_root
        self.train_box_model_dir = train_box_model_dir
        self.train_cyl_model_dir = train_cyl_model_dir
        self.train_tri_model_dir = train_tri_model_dir
        self._train_model_dirs = {
            "box": self.train_box_model_dir,
            "cyl": self.train_cyl_model_dir,
            "tri": self.train_tri_model_dir,
        }
        
        # 障碍物相关参数
        self.arena_half_size = arena_half_size
        self.wall_margin = wall_margin
        self.obstacle_min_dist_to_robot = obstacle_min_dist_to_robot
        self.obstacle_min_dist_to_goal = obstacle_min_dist_to_goal
        self.obstacle_min_dist_between = obstacle_min_dist_between
        self.num_train_obstacles = num_train_obstacles
        self.debug_obstacles = debug_obstacles
        self.obstacle_move_every_steps = obstacle_move_every_steps
        self.obstacle_move_jitter = obstacle_move_jitter
        self.obstacle_move_max_trials = obstacle_move_max_trials
        
        self.enable_viz = enable_viz
        self.viz_frame = viz_frame
        self.max_path_len = max_path_len
        
        # 训练模式标志位
        self.is_training = is_training
        
        # 训练专用：生成参数
        self.train_goal_range = 1.8 
        self.min_spawn_dist = 0.5 

        # 默认目标点
        if goal_x is None:
            goal_x = rospy.get_param("~goal_x", 1.0)
        if goal_y is None:
            goal_y = rospy.get_param("~goal_y", 0.0)
        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)

        # ---------------------------------------------------------------------
        # 定义 Gym 空间
        # ---------------------------------------------------------------------
        # 动作: 0=左转, 1=右转, 2=直行
        self.action_space = spaces.Discrete(3)
        # 观测: 90(雷达) + 1(相对角度) + 1(相对距离) + 1(上一步动作) = 93维
        obs_low = np.array([0.0] * 90 + [-1.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0] * 90 + [1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ---------------------------------------------------------------------
        # 初始化 ROS 通讯
        # ---------------------------------------------------------------------
        self._cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        
        self._current_scan: Optional[LaserScan] = None
        self._current_odom: Optional[Odometry] = None

        # 使用回调函数异步接收数据，避免阻塞
        self._scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self._scan_cb, queue_size=1)
        self._odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)

        # 缓存 Gazebo 服务代理（避免频繁创建）
        self._srv_set_state = None
        self._srv_spawn = None
        self._srv_delete = None
        if _HAS_MODEL_STATE:
            self._srv_set_state = rospy.ServiceProxy(self.set_model_state_service, SetModelState)
        if _HAS_MODEL_SPAWN:
            self._srv_spawn = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            self._srv_delete = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)

        # 初始化可视化 Publisher
        if self.enable_viz:
            self._waypoints_pub = rospy.Publisher("/kfdqn_viz/waypoints", MarkerArray, queue_size=1, latch=True)
            self._current_wp_pub = rospy.Publisher("/kfdqn_viz/current_wp", Marker, queue_size=1)
            self._trajectory_pub = rospy.Publisher("/kfdqn_viz/trajectory", Path, queue_size=1)

        self._np_random = np.random.default_rng()
        self.prev_action = 0.0
        self.prev_dis = 0.0
        self.step_count = 0
        self.goal = np.array([self.goal_x, self.goal_y], dtype=np.float32)
        self.path_msg: Optional[Path] = None
        self._sdf_cache: dict[str, str] = {}
        self._train_obs_names: list[str] = []
        self._train_obs_state: dict[str, dict] = {}
        self._episode_uid = 0
        self._last_obstacle_move_step = 0
        self._train_obstacles_spawned = False
        self.train_obstacle_z = 0.175
        self._train_fixed_names = [
            f"kfdqn_obs_train_{i}" for i in range(self.num_train_obstacles)
        ]

    # --- ROS 回调函数 ---
    def _scan_cb(self, msg: LaserScan):
        self._current_scan = msg

    def _odom_cb(self, msg: Odometry):
        self._current_odom = msg

    # --- 辅助函数：读取 SDF 模型文件 ---
    def _load_model_sdf(self, model_dir: str) -> str:
        cached = self._sdf_cache.get(model_dir)
        if cached is not None:
            return cached
        sdf_path = os.path.join(self.obstacle_models_root, model_dir, "model.sdf")
        try:
            with open(sdf_path, "r", encoding="utf-8") as sdf_file:
                sdf_content = sdf_file.read()
        except OSError as exc:
            raise RuntimeError(f"Failed to read SDF from {sdf_path}: {exc}") from exc
        self._sdf_cache[model_dir] = sdf_content
        return sdf_content

    def _load_sdf_from_model_dir(self, model_dir_name: str) -> str:
        return self._load_model_sdf(model_dir_name)

    # --- 障碍物管理逻辑 ---
    def _arena_limit(self) -> float:
        return self.arena_half_size - self.wall_margin

    def _sample_xy_yaw(self, goal_x, goal_y, existing_xy) -> tuple | None:
        """随机采样一个合法的障碍物坐标"""
        limit = self._arena_limit()
        for _ in range(300):
            x = float(self._np_random.uniform(-limit, limit))
            y = float(self._np_random.uniform(-limit, limit))
            # 距离检查：不能离机器人、目标或其他障碍物太近
            if math.hypot(x, y) < self.obstacle_min_dist_to_robot: continue
            if math.hypot(x - goal_x, y - goal_y) < self.obstacle_min_dist_to_goal: continue
            too_close = False
            for ex, ey in existing_xy:
                if math.hypot(x - ex, y - ey) < self.obstacle_min_dist_between:
                    too_close = True; break
            if too_close: continue
            yaw = float(self._np_random.uniform(-math.pi, math.pi))
            return x, y, yaw
        return None

    def _sample_train_obstacles(self, goal_x: float, goal_y: float) -> list[dict]:
        """
        [修改版] 生成位于起点和目标点中间的障碍物
        逻辑：在 Start->Goal 的连线上插值，并施加随机扰动
        """
        obstacles: list[dict] = []
        existing_xy: list[tuple[float, float]] = []
        
        # 强制生成 2 个障碍物 (或者使用 self.num_train_obstacles)
        target_obs_count = 2 
        
        # 获取起点坐标 (训练模式下 init_x/y 通常是 0,0，但为了通用性读取变量)
        start_x, start_y = self.init_x, self.init_y
        
        # 计算路径向量
        vec_x = goal_x - start_x
        vec_y = goal_y - start_y
        path_length = math.hypot(vec_x, vec_y)
        path_yaw = math.atan2(vec_y, vec_x)
        
        # 如果距离太近（比如生成的 Goal 离 Start 只有 0.5米），就不生成障碍物了，防止卡死
        if path_length < 1.0:
            return []

        for i in range(target_obs_count):
            # 计算插值比例：
            # 第一个障碍物在 33% 处，第二个在 66% 处
            # 增加一个 -0.1 到 0.1 的随机前后浮动
            ratio = ((i + 1) / (target_obs_count + 1)) + self._np_random.uniform(-0.1, 0.1)
            
            # 计算基准点 (在连线上)
            base_x = start_x + ratio * vec_x
            base_y = start_y + ratio * vec_y
            
            # 尝试生成合法坐标 (增加重试机制)
            valid_sample = False
            for _ in range(50): # 尝试 50 次微调
                # 添加 垂直于路径方向 的横向扰动 (Lateral Noise)
                # 范围设为 [-0.8, 0.8] 米，让它可能堵在路中间，也可能偏向一边留出通道
                lateral_offset = self._np_random.uniform(-0.8, 0.8)
                perp_yaw = path_yaw + math.pi / 2
                
                # 添加少量的 随机全向扰动 (Random Noise)
                rand_x = self._np_random.uniform(-0.2, 0.2)
                rand_y = self._np_random.uniform(-0.2, 0.2)
                
                # 最终候选坐标
                cand_x = base_x + lateral_offset * math.cos(perp_yaw) + rand_x
                cand_y = base_y + lateral_offset * math.sin(perp_yaw) + rand_y
                
                # 检查合法性 (复用原有的检查函数)
                if self._is_valid_obstacle_xy(cand_x, cand_y, goal_x, goal_y, existing_xy):
                    obs_type = str(self._np_random.choice(["box", "cyl", "tri"]))
                    yaw = float(self._np_random.uniform(-math.pi, math.pi))
                    
                    obstacles.append({"type": obs_type, "x": cand_x, "y": cand_y, "yaw": yaw})
                    existing_xy.append((cand_x, cand_y))
                    valid_sample = True
                    break
            
            # 如果尝试多次都无法在理想位置生成 (极少情况)，则跳过该障碍物
            if not valid_sample:
                pass 

        return obstacles

    def _is_valid_obstacle_xy(self, x, y, goal_x, goal_y, other_xy) -> bool:
        """验证一个障碍物坐标是否合法（移动时使用）"""
        limit = self._arena_limit()
        if abs(x) > limit or abs(y) > limit: return False
        if math.hypot(x, y) < self.obstacle_min_dist_to_robot: return False
        if math.hypot(x - goal_x, y - goal_y) < self.obstacle_min_dist_to_goal: return False
        for ox, oy in other_xy:
            if math.hypot(x - ox, y - oy) < self.obstacle_min_dist_between: return False
        return True

    def _propose_moved_xy(self, cur_x, cur_y) -> tuple:
        """计算障碍物随机移动后的新坐标"""
        jitter = self.obstacle_move_jitter
        dx = float(self._np_random.uniform(-jitter, jitter))
        dy = float(self._np_random.uniform(-jitter, jitter))
        limit = self._arena_limit()
        new_x = max(-limit, min(limit, cur_x + dx))
        new_y = max(-limit, min(limit, cur_y + dy))
        return new_x, new_y

    def _move_one_train_obstacle(self, name, new_x, new_y, yaw) -> bool:
        """调用 Gazebo 服务移动单个障碍物"""
        if not self._move_obstacle(name, new_x, new_y, yaw): return False
        if name in self._train_obs_state:
            self._train_obs_state[name]["x"] = float(new_x)
            self._train_obs_state[name]["y"] = float(new_y)
            self._train_obs_state[name]["yaw"] = float(yaw)
        return True

    def _maybe_move_train_obstacles(self) -> None:
        """尝试移动训练场景中的障碍物（增加环境动态性）"""
        if not (self.is_training and self.obstacle_mode and self.obstacle_move_every_steps > 0): return
        if self.step_count <= 0 or (self.step_count % self.obstacle_move_every_steps) != 0: return
        # ... (移动逻辑，包含重试机制) ...
        # [代码保持原样，仅做注释说明：这里会遍历所有障碍物，尝试微调其位置]
        try:
            goal_x = self.goal_x
            goal_y = self.goal_y
            moved = 0
            warned = False
            for name in list(self._train_obs_names):
                try:
                    state = self._train_obs_state.get(name)
                    if not state: continue
                    cur_x = float(state.get("x", 0.0))
                    cur_y = float(state.get("y", 0.0))
                    yaw = float(state.get("yaw", 0.0))
                    other_xy = [
                        (float(self._train_obs_state[other]["x"]), float(self._train_obs_state[other]["y"]))
                        for other in self._train_obs_names
                        if other != name and other in self._train_obs_state
                    ]
                    for _ in range(self.obstacle_move_max_trials):
                        nx, ny = self._propose_moved_xy(cur_x, cur_y)
                        if self._is_valid_obstacle_xy(nx, ny, goal_x, goal_y, other_xy):
                            if self._move_one_train_obstacle(name, nx, ny, yaw):
                                moved += 1
                            break
                except Exception as exc:
                    if self.debug_obstacles and not warned:
                        print(f"[train_obstacles] move warning: {exc}")
                        warned = True
                    continue
            if self.debug_obstacles:
                print(f"[train_obstacles] moved={moved}")
        except Exception as exc:
            if self.debug_obstacles:
                print(f"[train_obstacles] move warning: {exc}")
            return

    def _clear_train_obstacles(self) -> None:
        """清除所有训练生成的障碍物"""
        for name in list(self._train_obs_names):
            self._delete_obstacle(name)
        self._train_obs_names.clear()
        self._train_obs_state.clear()

    def _ensure_train_obstacles_spawned(self) -> None:
        if self._train_obstacles_spawned:
            return
        self._train_obs_names.clear()
        self._train_obs_state.clear()
        for name in self._train_fixed_names:
            if not self._spawn_one_train_obstacle(name, "box", 100.0, 100.0, 0.0):
                raise RuntimeError(f"Failed to spawn train obstacle: {name}")
        self._train_obstacles_spawned = True

    def _spawn_one_train_obstacle(self, name, obs_type, x, y, yaw) -> bool:
        """生成单个训练障碍物"""
        model_dir = self._train_model_dirs[obs_type]
        if self._spawn_model_from_dir(name, model_dir, x=x, y=y, z=self.train_obstacle_z, yaw=yaw):
            if name not in self._train_obs_names:
                self._train_obs_names.append(name)
            self._train_obs_state[name] = {"type": obs_type, "x": float(x), "y": float(y), "yaw": float(yaw)}
            return True
        return False

    # --- Gazebo 服务调用封装 ---

    def _call_reset(self) -> None:
        """
        复位 Gazebo 世界/仿真。
        优先调用 reset_world_service，失败再调用 reset_sim_service。
        """
        # 先尝试 reset_world（推荐：只重置世界状态，不重置时间）
        try:
            rospy.wait_for_service(self.reset_world_service, timeout=self.wait_timeout)
            rospy.ServiceProxy(self.reset_world_service, Empty)()
            return
        except Exception:
            pass

        # 再尝试 reset_simulation（更强：可能重置仿真时间等）
        try:
            rospy.wait_for_service(self.reset_sim_service, timeout=self.wait_timeout)
            rospy.ServiceProxy(self.reset_sim_service, Empty)()
        except Exception:
            # 这里不 raise，避免训练直接崩；但你可以在 debug 时打印
            if getattr(self, "debug_obstacles", False):
                print(f"[reset] failed to call {self.reset_world_service} and {self.reset_sim_service}")

    def _call_set_model_state(self) -> None:
        """
        使用 /gazebo/set_model_state 将机器人传送到 (init_x, init_y, init_yaw) 并清零速度。
        注意：需要 robot_model_name 和 gazebo set_model_state 可用。
        """
        if (self.robot_model_name is None) or (not _HAS_MODEL_STATE):
            return

        try:
            rospy.wait_for_service(self.set_model_state_service, timeout=self.wait_timeout)
            srv = self._srv_set_state
            if srv is None:
                return
            state = ModelState()
            state.model_name = self.robot_model_name

            # 位置
            state.pose.position.x = float(self.init_x)
            state.pose.position.y = float(self.init_y)
            # 机器人底盘离地略抬高，避免穿地/碰撞接触异常
            state.pose.position.z = 0.03

            # 朝向（只设置 yaw）
            yaw = float(self.init_yaw)
            state.pose.orientation.z = math.sin(yaw / 2.0)
            state.pose.orientation.w = math.cos(yaw / 2.0)

            # 清零速度
            state.twist.linear.x = 0.0
            state.twist.linear.y = 0.0
            state.twist.linear.z = 0.0
            state.twist.angular.x = 0.0
            state.twist.angular.y = 0.0
            state.twist.angular.z = 0.0

            srv(state)
        except Exception as exc:
            if getattr(self, "debug_obstacles", False):
                print(f"[set_model_state] failed: {exc}")

    def _spawn_model_from_dir(self, model_name, model_dir, x=0.0, y=0.0, z=0.0, yaw=0.0) -> bool:
        sdf_xml = self._load_model_sdf(model_dir)
        return self._call_spawn_sdf_model(model_name, sdf_xml, x, y, z, yaw)

    def _spawn_obstacle(self, name, model_dir_name, x, y, yaw) -> bool:
        """评估模式专用：生成固定障碍物"""
        sdf_xml = self._load_model_sdf(model_dir_name)
        return self._call_spawn_sdf_model(name, sdf_xml, x, y, 0.0, yaw)

    def _delete_obstacle(self, name: str) -> bool:
        if not _HAS_MODEL_SPAWN: return False
        try:
            rospy.wait_for_service("/gazebo/delete_model", timeout=self.wait_timeout)
            srv = self._srv_delete
            if srv is None:
                return False
            resp = srv(name)
            return bool(getattr(resp, "success", True))
        except Exception: return False

    def _move_obstacle(self, name, x, y, yaw) -> bool:
        """使用 set_model_state 服务移动模型"""
        if not _HAS_MODEL_STATE: return False
        try:
            rospy.wait_for_service(self.set_model_state_service, timeout=self.wait_timeout)
            srv = self._srv_set_state
            if srv is None:
                return False
            state = ModelState()
            state.model_name = name
            state.pose.position.x = float(x); state.pose.position.y = float(y); state.pose.position.z = self.train_obstacle_z
            state.pose.orientation.z = math.sin(yaw / 2.0); state.pose.orientation.w = math.cos(yaw / 2.0)
            state.twist.linear.x = 0.0; state.twist.linear.y = 0.0; state.twist.linear.z = 0.0
            state.twist.angular.x = 0.0; state.twist.angular.y = 0.0; state.twist.angular.z = 0.0
            resp = srv(state)
            return bool(getattr(resp, "success", True))
        except Exception: return False

    def _call_spawn_sdf_model(self, model_name, sdf_xml, x, y, z, yaw) -> bool:
        if not _HAS_MODEL_SPAWN: return False
        pose = Pose()
        pose.position.x = float(x); pose.position.y = float(y); pose.position.z = float(z)
        pose.orientation.z = math.sin(yaw / 2.0); pose.orientation.w = math.cos(yaw / 2.0)
        try:
            rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=self.wait_timeout)
            srv = self._srv_spawn
            if srv is None:
                return False
            resp = srv(model_name, sdf_xml, "", pose, "world")
            return bool(getattr(resp, "success", True))
        except Exception: return False

    def _publish_cmd(self, linear_x: float, angular_z: float) -> None:
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self._cmd_pub.publish(cmd)
        
    def _generate_training_setup(self):
        """训练模式专用：随机生成起点和终点"""
        # 1. 固定机器人出生点为原点 (0,0)
        self.init_x = 0.0
        self.init_y = 0.0
        self.init_yaw = float(self._np_random.uniform(-math.pi, math.pi))

        
        # 2. 随机生成目标点
        while True:
            self.goal_x = float(self._np_random.uniform(-self.train_goal_range, self.train_goal_range))
            self.goal_y = float(self._np_random.uniform(-self.train_goal_range, self.train_goal_range))
            
            # 距离检查：确保目标点离原点足够远
            dist = math.hypot(self.goal_x - self.init_x, self.goal_y - self.init_y)
            if dist > self.min_spawn_dist:
                break

    def _get_pose(self, odom: Odometry) -> Tuple[float, float, float]:
        pos = odom.pose.pose.position
        ori = odom.pose.pose.orientation
        yaw = _yaw_from_quat(ori.x, ori.y, ori.z, ori.w)
        return float(pos.x), float(pos.y), float(yaw)

    def current_goal(self) -> Tuple[float, float]:
        """获取当前目标点坐标"""
        # 训练模式下，忽略 Waypoints，直接返回随机生成的 goal
        if self.is_training or self.waypoints is None:
            return float(self.goal_x), float(self.goal_y)
        # 评估模式下，返回当前 Waypoint
        return self.waypoints[self.wp_idx]

    def _lidar90(self, scan: LaserScan) -> np.ndarray:
        """将 360 度雷达数据降采样为 90 维"""
        if scan.range_max > 0.0:
            self.max_lidar_range = float(scan.range_max)

        ranges = np.array(list(scan.ranges), dtype=np.float32)
        if ranges.size == 0:
            return np.full(90, self.max_lidar_range, dtype=np.float32)
        
        n = ranges.size
        bins = np.full(90, self.max_lidar_range, dtype=np.float32)
        for i in range(90):
            start = int(i * n / 90)
            end = int((i + 1) * n / 90)
            if end <= start:
                segment = ranges[min(start, n - 1) : min(start, n - 1) + 1]
            else:
                segment = ranges[start:end]
            finite = segment[np.isfinite(segment)]
            if finite.size == 0:
                bins[i] = self.max_lidar_range
            else:
                bins[i] = float(np.min(finite))
        
        bins = np.clip(bins, 0.0, self.max_lidar_range)
        return bins.astype(np.float32)

    def _normalize_obs(self, lidar: np.ndarray, theta_d: float, dis: float, prev_action: float) -> np.ndarray:
        """归一化观测向量"""
        lidar_norm = lidar / self.max_lidar_range
        theta_norm = theta_d / math.pi
        # 归一化距离：训练时用固定范围 (5.0)，评估时用最大距离 (8.0)
        norm_dist = 5.0 if self.is_training else self.max_goal_distance 
        dis_norm = np.clip(dis / norm_dist, 0.0, 1.0)
        return np.concatenate([lidar_norm, np.array([theta_norm, dis_norm, prev_action], dtype=np.float32)], axis=0)

    # -------------------------------------------------------------------------
    # 可视化 Marker 发布
    # -------------------------------------------------------------------------
    def _publish_waypoints_markers(self) -> None:
        if not self.enable_viz: return
        
        # 训练模式只显示当前单一目标，评估模式显示完整路径点
        if self.is_training:
            points = [(self.goal_x, self.goal_y)]
        else:
            points = self.waypoints if self.waypoints else [(self.goal_x, self.goal_y)]
        
        marker_array = MarkerArray()
        for idx, (gx, gy) in enumerate(points):
            marker = Marker()
            marker.header.frame_id = self.viz_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = gx
            marker.pose.position.y = gy
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.12; marker.scale.y = 0.12; marker.scale.z = 0.12
            marker.color.b = 1.0; marker.color.a = 0.6
            marker_array.markers.append(marker)
        self._waypoints_pub.publish(marker_array)

    def _publish_current_wp_marker(self) -> None:
        if not self.enable_viz: return
        gx, gy = self.current_goal()
        
        marker = Marker()
        marker.header.frame_id = self.viz_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "current_wp"
        marker.id = 999
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = gx; marker.pose.position.y = gy; marker.pose.position.z = 0.06
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.18; marker.scale.y = 0.18; marker.scale.z = 0.18
        marker.color.r = 1.0; marker.color.a = 1.0
        self._current_wp_pub.publish(marker)

    # -------------------------------------------------------------------------
    # Gym 接口: Reset
    # -------------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            
        # 安全重置循环：确保机器人不会出生在墙里
        max_retries = 100 
        for _ in range(max_retries):
            # 1. 模式选择与坐标生成
            if self.is_training:
                # 训练模式：固定原点，随机目标
                self._generate_training_setup()
                self.waypoints = None 
                self.wp_idx = 0
            else:
                # 评估模式：固定路径 (Waypoints)，固定起点
                self.wp_idx = 0
                self.init_x = 0.0 
                self.init_y = 0.0
            
            # 2. 物理重置 + 传送
            self._current_scan = None
            self._current_odom = None
            
            odom_seq0 = self._current_odom.header.seq if self._current_odom else -1
            scan_seq0 = self._current_scan.header.seq if self._current_scan else -1

            self._call_reset()
            self._call_set_model_state()
            self._publish_cmd(0.0, 0.0)

            # --- 训练模式：动态障碍物生成 ---
            if self.is_training and self.obstacle_mode:
                self._ensure_train_obstacles_spawned()
                attempts = 0
                last_sample_count = 0
                last_spawn_success = 0
                last_min_lidar = float("inf")
                while True:
                    obs_list = self._sample_train_obstacles(self.goal_x, self.goal_y)
                    last_sample_count = len(obs_list)
                    last_spawn_success = 0
                    for i in range(self.num_train_obstacles):
                        name = self._train_fixed_names[i]
                        if i < len(obs_list):
                            obs = obs_list[i]
                            x = obs["x"]
                            y = obs["y"]
                            yaw = obs["yaw"]
                            obs_type = obs["type"]
                        else:
                            x = 100.0
                            y = 100.0
                            yaw = 0.0
                            obs_type = self._train_obs_state.get(name, {}).get("type", "box")
                        if self._move_obstacle(name, x, y, yaw):
                            if i < len(obs_list):
                                last_spawn_success += 1
                            self._train_obs_state[name] = {
                                "type": obs_type,
                                "x": float(x),
                                "y": float(y),
                                "yaw": float(yaw),
                            }
                    
                    # 等待雷达数据更新以确认生成安全
                    scan_seq = self._current_scan.header.seq if self._current_scan else -1
                    scan_msg = None
                    start_wait_scan = time.time()
                    while time.time() - start_wait_scan < 0.5:
                        scan_msg = self._current_scan
                        if scan_msg and scan_msg.header.seq > scan_seq: break
                        time.sleep(0.005)
                    
                    if scan_msg is None: scan_msg = self._current_scan
                    min_lidar = float("inf")
                    if scan_msg is not None: min_lidar = float(np.min(self._lidar90(scan_msg)))
                    last_min_lidar = min_lidar
                    
                    # 如果生成后离机器人太近，重试
                    if min_lidar < self.CTH + 0.05 and attempts < 4:
                        attempts += 1
                        continue
                    break
                if self.debug_obstacles:
                    print(f"[train_obstacles] sampled={last_sample_count} spawned={last_spawn_success} min_lidar={last_min_lidar:.3f} retries={attempts}")

            # --- 评估模式：固定障碍物生成 ---
            if (not self.is_training) and self.obstacle_mode:
                self._delete_obstacle("box_a")
                self._delete_obstacle("box_b")
                # 评估模式下生成的障碍物位置是固定的
                if not self._spawn_model_from_dir("box_a", "box_a"):
                    raise RuntimeError("Failed to spawn eval obstacle: box_a")
                if not self._spawn_model_from_dir("box_b", "box_b"):
                    raise RuntimeError("Failed to spawn eval obstacle: box_b")
            
            # 3. 等待数据归位 (同步等待)
            start_wait = time.time()
            data_valid = False
            
            while True:
                scan = self._current_scan
                odom = self._current_odom
                if scan and odom:
                    # 确保是 reset 之后的新数据
                    if scan.header.seq > scan_seq0 and odom.header.seq > odom_seq0:
                        x, y, yaw = self._get_pose(odom)
                        # 检查是否真的传送到了指定位置
                        if (x - self.init_x) ** 2 + (y - self.init_y) ** 2 < (0.15 ** 2):
                            data_valid = True
                            break
                if time.time() - start_wait > 2.0: break 
                time.sleep(0.005)
            
            if not data_valid: continue

            # 4. [核心修复] 检查出生点安全性
            lidar_check = self._lidar90(self._current_scan)
            if np.min(lidar_check) < self.CTH + 0.05:
                if self.is_training: continue
                # 评估模式下通常不重随，但会警告
            
            break

        if self.enable_viz:
            self.path_msg = Path()
            self.path_msg.header.frame_id = self.viz_frame
            self._trajectory_pub.publish(self.path_msg) 

        self.step_count = 0
        self.prev_action = 0.0
        
        gx, gy = self.current_goal()
        self.goal = np.array([gx, gy], dtype=np.float32)

        # 随机目标逻辑（仅兼容旧配置）
        if not self.is_training and self.waypoints is None and self.random_goal:
            angle = float(self._np_random.uniform(-math.pi, math.pi))
            radius = float(self._np_random.uniform(0.5, self.max_goal_distance))
            self.goal_x = float(radius * math.cos(angle))
            self.goal_y = float(radius * math.sin(angle))
            self.goal = np.array([self.goal_x, self.goal_y], dtype=np.float32)

        if self.enable_viz:
            self._publish_waypoints_markers()
            self._publish_current_wp_marker()

        lidar = self._lidar90(self._current_scan)
        x, y, yaw = self._get_pose(self._current_odom)
        
        gx, gy = self.current_goal()
        dx = gx - x
        dy = gy - y
        dis = float(math.hypot(dx, dy))
        theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)
        self.prev_dis = dis

        obs = self._normalize_obs(lidar, theta_d, dis, self.prev_action)
        info = {"min_lidar": float(np.min(lidar)), "theta_d": float(theta_d), "dis": float(dis)}
        return obs.astype(np.float32), info

    # -------------------------------------------------------------------------
    # Gym Step
    # -------------------------------------------------------------------------
    def step(self, action: int):
        assert self.action_space.contains(action)
        self.step_count += 1

        # 动作指令转换
        if action == self.ACTION_LEFT:
            lin, ang = self.turn_v, self.turn_omega
        elif action == self.ACTION_RIGHT:
            lin, ang = self.turn_v, -self.turn_omega
        else:
            lin, ang = self.forward_v, 0.0

        # # 动态障碍物移动
        # self._maybe_move_train_obstacles()

        scan_seq0 = self._current_scan.header.seq if self._current_scan else -1

        # 执行动作 (持续 action_duration)
        start_time = rospy.get_time()
        end_time = start_time + self.action_duration
        rate = rospy.Rate(self.publish_hz)

        while rospy.get_time() < end_time:
            if rospy.is_shutdown(): break
            try:
                self._publish_cmd(lin, ang)
                rate.sleep()
            except rospy.ROSTimeMovedBackwardsException: break
        
        # 等待下一帧观测数据
        t0 = rospy.get_time()
        timeout = 0.7 
        rate_wait = rospy.Rate(120)

        while True:
            scan = self._current_scan
            odom = self._current_odom
            if scan and odom and scan.header.seq > scan_seq0:
                break
            if rospy.get_time() - t0 > timeout: break
            rate_wait.sleep()
            
        if scan is None or odom is None:
             raise RuntimeError("Data loss during step.")

        # 可视化轨迹更新
        if self.enable_viz:
            if self.path_msg is None:
                self.path_msg = Path()
                self.path_msg.header.frame_id = self.viz_frame
            pose = PoseStamped()
            pose.header.frame_id = self.viz_frame
            pose.header.stamp = rospy.Time.now()
            pose.pose = odom.pose.pose
            self.path_msg.poses.append(pose)
            if len(self.path_msg.poses) > self.max_path_len:
                self.path_msg.poses = self.path_msg.poses[-self.max_path_len :]
            self._trajectory_pub.publish(self.path_msg)

        # 状态计算
        lidar = self._lidar90(scan)
        min_lidar = float(np.min(lidar))
        x, y, yaw = self._get_pose(odom)
        gx, gy = self.current_goal()
        dx = gx - x
        dy = gy - y
        dis = float(math.hypot(dx, dy))
        theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)

        terminated = False
        truncated = False
        info = {"min_lidar": float(min_lidar), "theta_d": float(theta_d), "dis": float(dis)}

        # --- 奖励计算逻辑 ---
        if min_lidar < self.CTH:
            # 碰撞
            reward = self.r_collision
            terminated = True
            info["is_collision"] = True
        else:
            # 引导奖励 (Shaping Reward)
            reward = (self.prev_dis - dis) * self.p_r + self.r_o
            
            if dis < self.waypoint_rth: 
                if self.is_training:
                    # 训练模式：到达随机目标即结束 (单目标)
                    reward = self.r_reach
                    terminated = True
                    info["is_success"] = True
                else:
                    # 评估模式：切换 Waypoint (多目标路径)
                    if self.waypoints is not None and self.wp_idx < len(self.waypoints) - 1:
                        self.wp_idx += 1
                        new_gx, new_gy = self.current_goal()
                        # 更新相对位置
                        dx = new_gx - x
                        dy = new_gy - y
                        dis = float(math.hypot(dx, dy))
                        theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)
                        self.prev_dis = dis 
                        
                        reward += self.r_reach # [用户指定] 到达中间点也给大奖励
                        info["waypoint_reached"] = True
                        info["wp_idx"] = self.wp_idx
                        if self.enable_viz:
                            self._publish_current_wp_marker()
                    else:
                        # 到达最后一个 Waypoint
                        reward = self.r_reach
                        terminated = True
                        info["is_success"] = True

        if self.step_count >= self.max_steps:
            truncated = True

        self.prev_action = float(action) / 2.0
        if not terminated:
            self.prev_dis = dis
        
        if terminated or truncated:
            self._publish_cmd(0.0, 0.0)

        obs = self._normalize_obs(lidar, theta_d, dis, self.prev_action)
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def close(self) -> None:
        self._publish_cmd(0.0, 0.0)

# --- 环境注册 ---

if "GoalReachTrain-v0" not in registry:
    register(
        id="GoalReachTrain-v0",
        entry_point="envs_ros.ros_gazebo_mobile_robot_env:ROSGazeboMobileRobotEnv",
        kwargs={
            "obstacle_mode": False,
            "robot_model_name": "turtlebot3_burger",
            "is_training": True, # 开启训练模式 (随机起点终点)
        },
    )
if "GoalReachEval-v0" not in registry:
    register(
        id="GoalReachEval-v0",
        entry_point="envs_ros.ros_gazebo_mobile_robot_env:ROSGazeboMobileRobotEnv",
        kwargs={
            "obstacle_mode": False,
            "robot_model_name": "turtlebot3_burger",
            "waypoints": [(0.7, 0.5), (1.6, 0.5), (1.3, -0.4), (0.8, 0.0), (0.8, -0.5)],
            "waypoint_rth": 0.20,
            "is_training": False, # 开启评估模式 (固定路径)
        },
    )

if "ObstacleAvoidTrain-v0" not in registry:
    register(
        id="ObstacleAvoidTrain-v0",
        entry_point="envs_ros.ros_gazebo_mobile_robot_env:ROSGazeboMobileRobotEnv",
        kwargs={
            "obstacle_mode": True,
            "robot_model_name": "turtlebot3_burger",
            "is_training": True,
        },
    )

if "ObstacleAvoidEval-v0" not in registry:
    register(
        id="ObstacleAvoidEval-v0",
        entry_point="envs_ros.ros_gazebo_mobile_robot_env:ROSGazeboMobileRobotEnv",
        kwargs={
            "obstacle_mode": True,
            "robot_model_name": "turtlebot3_burger",
            "waypoints": [(1.4, 0.9), (-0.15, 0.1)],
            "waypoint_rth": 0.20,
            "is_training": False,
        },
    )
