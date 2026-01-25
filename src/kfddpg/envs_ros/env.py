from __future__ import annotations

import math
import time
from typing import Optional, Tuple
import pathlib
import csv
import gymnasium as gym
import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped, Twist
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker

# --- 检查 Gazebo 服务依赖 ---
try:
    from gazebo_msgs.srv import SetModelState
    from gazebo_msgs.msg import ModelState
    _HAS_MODEL_STATE = True
except Exception:
    _HAS_MODEL_STATE = False

_ROS_INITIALIZED = False

def ensure_ros_init(node_name: str = "gym_ddpg_turtlebot") -> None:
    global _ROS_INITIALIZED
    if _ROS_INITIALIZED or rospy.core.is_initialized():
        return
    rospy.init_node(node_name, anonymous=True, disable_signals=True)
    _ROS_INITIALIZED = True

def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi

def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - (2.0 * (y * y + z * z))
    return math.atan2(siny, cosy)

class ROSGazeboMobileRobotEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        # --- ROS 配置 (保持不变) ---
        scan_topic: str = "/scan",
        odom_topic: str = "/odom",
        cmd_vel_topic: str = "/cmd_vel",
        reset_world_service: str = "/gazebo/reset_world",
        reset_sim_service: str = "/gazebo/reset_simulation",
        set_model_state_service: str = "/gazebo/set_model_state",
        robot_model_name: str | None = None,

        # --- 物理参数 (修改默认值为 Turtlebot3 极限) ---
        max_steps: int = 200,          # DDPG 通常步数多一点比较好
        max_lidar_range: float = 3.5,
        
        # [新增] 物理极限参数，用于 DDPG 动作缩放
        max_linear_vel: float = 0.22,
        max_angular_vel: float = 1,

        publish_hz: float = 100.0,
        action_duration: float = 0.2, # 动作持续时间，近似控制频率 1/f

        # --- 奖励参数 (保持不变) ---
        RTH: float = 0.20,
        CTH: float = 0.15,
        r_reach: float = 200.0,
        r_collision: float = -200.0,
        p_r: float = 50,
        r_o: float = -1,

        # --- 其他配置 (保持不变) ---
        waypoint_rth: float = 0.20,
        wait_timeout: float = 1.0,
        obstacle_mode: bool = False,
        debug_obstacles: bool = False,
        map_xy_limit: float = 2.0,
        wall_margin: float = 0.4,
        goal_d_min: float = 0.5,
        goal_d_max: float = 6.5,
        safety_margin: float = 0.05,
        max_reset_retries: int = 150,
        continue_on_success: bool = False,
        enable_viz: bool = True, # [可选] 如果不想看 Rviz 轨迹，设为 False
        viz_frame: str = "odom",
        max_path_len: int = 500,
        render_mode: str | None = None,
    ):
        super().__init__()
        ensure_ros_init()

        # 参数赋值
        self.scan_topic = scan_topic
        self.odom_topic = odom_topic
        self.cmd_vel_topic = cmd_vel_topic
        self.reset_world_service = reset_world_service
        self.reset_sim_service = reset_sim_service
        self.set_model_state_service = set_model_state_service
        self.robot_model_name = robot_model_name
        self.max_steps = max_steps
        self.max_lidar_range = max_lidar_range
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.publish_hz = publish_hz
        self.action_duration = action_duration
        self.RTH = RTH
        self.CTH = CTH
        self.r_reach = r_reach
        self.r_collision = r_collision
        self.p_r = p_r
        self.r_o = r_o
        self.waypoint_rth = waypoint_rth
        self.goal_d_max = goal_d_max
        self.wait_timeout = wait_timeout
        self.obstacle_mode = obstacle_mode
        self.debug_obstacles = debug_obstacles
        self.map_xy_limit = float(map_xy_limit)
        self.wall_margin = float(wall_margin)
        self.goal_d_min = float(goal_d_min)
        self.goal_d_max = float(goal_d_max)
        self.safety_margin = float(safety_margin)
        self.max_reset_retries = int(max_reset_retries)
        self.continue_on_success = bool(continue_on_success)
        self.enable_viz = enable_viz
        self.viz_frame = viz_frame
        self.max_path_len = max_path_len

        # --- [关键修改 1] 动作空间改为连续 ---
        # 即使 Actor 输出是 Sigmoid/Tanh，环境层最好也定义清楚物理边界
        # 形状: (2,) -> [linear_vel, angular_vel]
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.max_angular_vel], dtype=np.float32),
            high=np.array([self.max_linear_vel, self.max_angular_vel], dtype=np.float32),
            dtype=np.float32
        )

        # --- [关键修改 2] 观察空间维度调整 ---
        # 90 (雷达) + 1 (角度差) + 1 (距离) + 2 (上一步动作 v, w) = 94 维
        # 注意：原代码是 +1 (上一步动作索引)，现在需要变成向量
        self.obs_dim = 94
        obs_low = np.array([0.0] * 90 + [-1.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs_high = np.array([1.0] * 90 + [1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ROS 通信
        self._cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self._current_scan: Optional[LaserScan] = None
        self._current_odom: Optional[Odometry] = None
        self._scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self._scan_cb, queue_size=1)
        self._odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)

        self._srv_set_state = None
        if _HAS_MODEL_STATE:
            self._srv_set_state = rospy.ServiceProxy(self.set_model_state_service, SetModelState)

        # 可视化 (可选功能 1)
        if self.enable_viz:
            self._current_wp_pub = rospy.Publisher("/ddpg_viz/current_wp", Marker, queue_size=1)
            self._trajectory_pub = rospy.Publisher("/ddpg_viz/trajectory", Path, queue_size=1)

        self._np_random = np.random.default_rng()
        
        # [修改] prev_action 初始化为 2维 0向量
        self.prev_action = np.zeros(2, dtype=np.float32) 
        
        self.prev_dis = 0.0
        self.step_count = 0
        self.init_x = 0.0
        self.init_y = 0.0
        self.init_yaw = 0.0
        self.goal = np.array([0.0, 0.0], dtype=np.float32)
        self.path_msg: Optional[Path] = None

        self.obstacles = [
            (-0.6, -0.6, 0.35), (-0.6, 0.6, 0.35), (0.6, -0.6, 0.35), (0.6, 0.6, 0.35),
            (1.7, 0.0, 0.35), (-1.7, 0.0, 0.35), (0.0, 1.7, 0.35), (0.0, -1.7, 0.35),
        ]

        # 固定目标 (可选功能 2)
        self.use_fixed_goal_list = False # 默认关掉，如果需要改为 True
        self.episode_count = 0
        self.goal_list_path = pathlib.Path(__file__).resolve().parent / "list_goal.csv"
        self._goal_list = None

    def _scan_cb(self, msg: LaserScan): self._current_scan = msg
    def _odom_cb(self, msg: Odometry): self._current_odom = msg

    # --- 这里是 Gazebo 强制重置服务，建议保留 ---
    def _call_reset(self) -> None:
        try:
            rospy.wait_for_service(self.reset_world_service, timeout=self.wait_timeout)
            rospy.ServiceProxy(self.reset_world_service, Empty)()
        except Exception: pass
        try:
            rospy.wait_for_service(self.reset_sim_service, timeout=self.wait_timeout)
            rospy.ServiceProxy(self.reset_sim_service, Empty)()
        except Exception: pass

    def _call_set_model_state(self) -> None:
        if (self.robot_model_name is None) or (not _HAS_MODEL_STATE): return
        try:
            rospy.wait_for_service(self.set_model_state_service, timeout=self.wait_timeout)
            state = ModelState()
            state.model_name = self.robot_model_name
            state.pose.position.x = float(self.init_x)
            state.pose.position.y = float(self.init_y)
            state.pose.position.z = 0.03
            yaw = float(self.init_yaw)
            state.pose.orientation.z = math.sin(yaw / 2.0)
            state.pose.orientation.w = math.cos(yaw / 2.0)
            state.twist.linear.x = state.twist.linear.y = state.twist.linear.z = 0.0
            state.twist.angular.x = state.twist.angular.y = state.twist.angular.z = 0.0
            self._srv_set_state(state)
        except Exception: pass

    def _publish_cmd(self, linear_x: float, angular_z: float) -> None:
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self._cmd_pub.publish(cmd)

    def _get_pose(self, odom: Odometry) -> Tuple[float, float, float]:
        pos = odom.pose.pose.position
        ori = odom.pose.pose.orientation
        yaw = _yaw_from_quat(ori.x, ori.y, ori.z, ori.w)
        return float(pos.x), float(pos.y), float(yaw)

    def _lidar90(self, scan: LaserScan) -> np.ndarray:
        # 1. 安全检查：如果 scan 还没收到数据，返回全量量程
        if scan is None or len(scan.ranges) == 0:
            return np.full(90, self.max_lidar_range, dtype=np.float32)

        # 2. 获取原始数据（此时 len(ranges) 应该是 90）
        ranges = np.array(list(scan.ranges), dtype=np.float32)
        
        # 3. 处理无效值（inf 和 nan）
        # 逻辑：inf 表示没扫到障碍物，填充为最大距离；nan 通常是物理噪声，填充为0
        ranges[~np.isfinite(ranges)] = self.max_lidar_range
        ranges = np.clip(ranges, 0.0, self.max_lidar_range)

        # 4. 关键适配逻辑：
        # 如果 Gazebo 已经是 90 维，直接返回；如果不是，自动重采样为 90 维。
        n = ranges.size
        if n == 90:
            return ranges.astype(np.float32)
        else:
            # 这是一个通用的下采样逻辑，确保输入维度不匹配时环境不会崩溃
            indices = (np.arange(90) * (n / 90.0)).astype(int)
            return ranges[indices].astype(np.float32)

    # 归一化逻辑适配 94 维 ---
    def _normalize_obs(self, lidar: np.ndarray, theta_d: float, dis: float, prev_action: np.ndarray) -> np.ndarray:
        lidar_norm = lidar / self.max_lidar_range
        theta_norm = theta_d / math.pi
        dis_norm = np.clip(dis / self.goal_d_max, 0.0, 1.0)
        
        # 归一化旧动作：将物理值转回 [0,1] 和 [-1,1] 区间，这有助于神经网络训练
        # 线速度 [0, 0.22] -> [0, 1]
        prev_v_norm = prev_action[0] / self.max_linear_vel
        # 角速度 [-2.84, 2.84] -> [-1, 1]
        prev_w_norm = prev_action[1] / self.max_angular_vel
        
        # 拼接顺序：90雷达 + 1角度 + 1距离 + 2动作 = 94
        return np.concatenate(
            [lidar_norm, 
             np.array([theta_norm, dis_norm, prev_v_norm, prev_w_norm], dtype=np.float32)], 
            axis=0
        )

    def _sample_uniform_xy(self) -> Tuple[float, float]:
        lim = self.map_xy_limit - self.wall_margin
        x = float(self._np_random.uniform(-lim, lim))
        y = float(self._np_random.uniform(-lim, lim))
        z = float(self._np_random.uniform(-np.pi, np.pi))
        return x, y, z

    def _sample_robot_pose(self) -> Tuple[float, float, float]:
        x, y, z= self._sample_uniform_xy()
        # 从原点出生，就保持 0,0,0
        return x, y, z 

    def _sample_goal(self, robot_x: float, robot_y: float) -> Tuple[float, float]:
        """
        [修改] 随机目标模式 (无障碍物版本)
        在 4x4 地图范围内随机生成目标点。
        """
        # 4x4 地图意味着 x, y 范围是 [-2, 2]
        # wall_margin 是为了不让目标生成在墙里面
        lim = self.map_xy_limit - self.wall_margin 

        for _ in range(100): # 尝试 100 次
            gx = float(self._np_random.uniform(-lim, lim))
            gy = float(self._np_random.uniform(-lim, lim))

            # 计算与机器人的距离
            d = float(math.hypot(gx - robot_x, gy - robot_y))

            # 约束条件：
            # 1. 距离必须大于最小距离 (goal_d_min)
            # 2. 距离必须小于最大距离 (goal_d_max)
            #    注意：在 4x4 地图中，对角线最大距离约为 5.6m。
            #    如果 goal_d_max 设得太小（比如 3），角落可能永远采样不到。
            #    建议根据训练阶段动态调整，或者直接设大一点。
            if self.goal_d_min <= d <= self.goal_d_max:
                return gx, gy
        
        # 如果采样失败（极少情况），返回一个默认安全点
        return 1.5, 0.0

    def _load_goal_list_if_needed(self):
        if self._goal_list is not None: return
        if not self.goal_list_path.exists(): raise FileNotFoundError("Goal list missing")
        goals = []
        with self.goal_list_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader: goals.append((float(row["goal_x"]), float(row["goal_y"])))
        self._goal_list = goals

    def _get_fixed_goal(self, episode_idx: int):
        self._load_goal_list_if_needed()
        i = episode_idx % len(self._goal_list)
        return self._goal_list[i]

    def _publish_current_wp_marker(self) -> None:
        if not self.enable_viz: return
        gx, gy = float(self.goal[0]), float(self.goal[1])
        marker = Marker()
        marker.header.frame_id = self.viz_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "current_wp"
        marker.id = 999
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = gx; marker.pose.position.y = gy; marker.pose.position.z = 0.06
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.color.r = 1.0; marker.color.a = 1.0
        self._current_wp_pub.publish(marker)

    # --- [关键修改 4] Step 函数适配连续输入 ---
    def step(self, action: np.ndarray):
        """
        DDPG 传入的 action 应该是 [linear_vel, angular_vel] 的物理数值
        """
        self.step_count += 1
        
        # 你的 Actor 网络已经输出了符合物理范围的值（Sigmoid*max, Tanh*max）
        # 但为了安全，这里再次做一次截断
        lin_vel = float(np.clip(action[0], 0.0, self.max_linear_vel))
        ang_vel = float(np.clip(action[1], -self.max_angular_vel, self.max_angular_vel))

        scan_seq0 = self._current_scan.header.seq if self._current_scan else -1

        # 持续发布指令一段时间
        start_time = rospy.get_time()
        end_time = start_time + self.action_duration
        rate = rospy.Rate(self.publish_hz)
        while rospy.get_time() < end_time:
            if rospy.is_shutdown(): break
            self._publish_cmd(lin_vel, ang_vel)
            rate.sleep()

        # 等待新一帧雷达数据 (这是你原代码的核心逻辑，保留)
        t0 = rospy.get_time()
        while True:
            if rospy.get_time() - t0 > self.action_duration: break
            if self._current_scan and self._current_scan.header.seq > scan_seq0: break
            time.sleep(0.005)

        # 状态计算
        lidar = self._lidar90(self._current_scan)
        min_lidar = float(np.min(lidar))
        x, y, yaw = self._get_pose(self._current_odom)
        gx, gy = float(self.goal[0]), float(self.goal[1])
        dx, dy = gx - x, gy - y
        dis = float(math.hypot(dx, dy))
        theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)

        # 奖励计算 (逻辑完全保持不变)
        terminated = truncated = False
        info = {
            "min_lidar": min_lidar,
            "theta_d": theta_d,
            "dis": dis,
            "goal": (gx, gy)
        }

        if min_lidar < self.CTH:
            reward = self.r_collision
            terminated = True
            info["is_collision"] = True
        else:
            reward = (self.prev_dis - dis) * self.p_r + self.r_o
            
            if dis < self.waypoint_rth:
                info["is_success"] = True
                if self.continue_on_success:
                    reward = self.r_reach
                    new_gx, new_gy = self._sample_goal(x, y)
                    self.goal = np.array([new_gx, new_gy], dtype=np.float32)
                    if self.enable_viz: self._publish_current_wp_marker()
                    dx, dy = new_gx - x, new_gy - y
                    dis = float(math.hypot(dx, dy))
                    theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)
                    self.prev_dis = dis
                else:
                    reward = self.r_reach
                    terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        # 更新历史动作 (变为向量)
        self.prev_action = np.array([lin_vel, ang_vel], dtype=np.float32)
        if not terminated:
            self.prev_dis = dis
        
        if terminated or truncated:
            self._publish_cmd(0.0, 0.0)

        obs = self._normalize_obs(lidar, theta_d, dis, self.prev_action)
        
        # 轨迹可视化更新
        if self.enable_viz:
            if self.path_msg is None:
                self.path_msg = Path()
                self.path_msg.header.frame_id = self.viz_frame
            pose = PoseStamped()
            pose.header.frame_id = self.viz_frame
            pose.header.stamp = rospy.Time.now()
            pose.pose = self._current_odom.pose.pose
            self.path_msg.poses.append(pose)
            if len(self.path_msg.poses) > self.max_path_len:
                self.path_msg.poses = self.path_msg.poses[-self.max_path_len:]
            self.path_msg.header.stamp = rospy.Time.now()
            self._trajectory_pub.publish(self.path_msg)

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None: self._np_random = np.random.default_rng(seed)

        # 重置逻辑保持不变，这部分处理 Gazebo 延迟非常稳健
        for _ in range(self.max_reset_retries):
            rx, ry, ryaw = self._sample_robot_pose()
            self._current_scan = self._current_odom = None
            
            self.init_x, self.init_y, self.init_yaw = rx, ry, ryaw
            self._call_set_model_state()
            self._publish_cmd(0.0, 0.0)

            # 等待数据同步
            start_wait = time.time()
            data_valid = False
            while time.time() - start_wait < 2.0:
                scan, odom = self._current_scan, self._current_odom
                if scan and odom:
                    x, y, _ = self._get_pose(odom)
                    if (x - rx)**2 + (y - ry)**2 < 0.04: 
                        data_valid = True
                        break
                time.sleep(0.01)
            
            if not data_valid: continue

            if np.min(self._lidar90(self._current_scan)) < self.CTH + self.safety_margin:
                continue

            if self.use_fixed_goal_list:
                gx, gy = self._get_fixed_goal(self.episode_count)
            else:
                gx, gy = self._sample_goal(rx, ry)
            self.goal = np.array([gx, gy], dtype=np.float32)
            if self.enable_viz: self._publish_current_wp_marker()
            break
        else:
            print("Warning: Reset failed to find safe spot, forcing 0,0")
            self.init_x, self.init_y = 0, 0
            self._call_set_model_state()

        if self.enable_viz:
            self.path_msg = Path()
            self.path_msg.header.frame_id = self.viz_frame
            self._trajectory_pub.publish(self.path_msg)

        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32) # 重置动作记录

        lidar = self._lidar90(self._current_scan)
        x, y, yaw = self._get_pose(self._current_odom)
        dx, dy = float(self.goal[0]) - x, float(self.goal[1]) - y
        dis = float(math.hypot(dx, dy))
        theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)
        self.prev_dis = dis

        obs = self._normalize_obs(lidar, theta_d, dis, self.prev_action)
        info = {
            "min_lidar": float(np.min(lidar)),
            "theta_d": theta_d,
            "dis": dis,
            "goal": (float(self.goal[0]), float(self.goal[1]))
        }
        self.episode_count += 1
        return obs.astype(np.float32), info

    def close(self) -> None:
        self._publish_cmd(0.0, 0.0)

# 注册环境
if "ENV" not in registry:
    register(
        id="ENV",
        entry_point="kfddpg.envs_ros.env:ROSGazeboMobileRobotEnv", 
        kwargs={
            "obstacle_mode": False,           # 无障碍模式
            "robot_model_name": "turtlebot3_burger",
            "max_steps": 200,                 # 最大步数
        },
    )