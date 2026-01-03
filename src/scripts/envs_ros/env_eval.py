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

# --- 检查 Gazebo 服务依赖：判断当前环境下 Gazebo 相关的消息和服务是否可用 ---
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
    """将角度归一化到 [-pi, pi] 范围，常用于计算航向角偏差"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """从四元数计算偏航角 (Yaw)，将 3D 朝向转换为平面旋转角度"""
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - (2.0 * (y * y + z * z))
    return math.atan2(siny, cosy)

class ROSGazeboMobileRobotEnv(gym.Env):
    """
    ROS Gazebo 移动机器人强化学习环境
    支持：
    1. 离散动作空间 (左转, 右转, 前进)
    2. 连续状态空间 (90维雷达 + 相对位置信息)
    3. 固定路径点顺序导航 (Waypoints)
    4. 固定障碍物场景 (box_a/box_b)
    5. Rviz 可视化集成 (轨迹、目标点标记)
    """
    metadata = {"render_modes": []}
    
    # 动作枚举定义
    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_FORWARD = 2

    def __init__(
        self,
        *,
        # --- ROS 话题配置 ---
        scan_topic: str = "/scan",            # 激光雷达话题
        odom_topic: str = "/odom",            # 里程计算法话题
        cmd_vel_topic: str = "/cmd_vel",      # 速度控制话题
        reset_world_service: str = "/gazebo/reset_world",     # 重置世界服务
        reset_sim_service: str = "/gazebo/reset_simulation",  # 重置仿真服务
        set_model_state_service: str = "/gazebo/set_model_state", # 设置模型位置服务
        robot_model_name: str | None = None,   # Gazebo 中的机器人模型名称
        
        # --- 机器人初始状态 ---
        init_x: float = 0.0,
        init_y: float = 0.0,
        init_yaw: float = 0.0,
        
        # --- 训练参数 ---
        max_steps: int = 100,          # 单回合最大允许步数
        max_lidar_range: float = 3.5,  # 雷达截断距离（超过此距离按最大值算）
        forward_v: float = 0.11,       # 直行时的线速度 (m/s)
        turn_v: float = 0.11,          # 转向时的线速度 (m/s)
        turn_omega: float = math.pi/2,       # 转向时的角速度 (rad/s)
        publish_hz: float = 50.0,      # 控制频率
        action_duration: float = 0.2,  # 每个动作执行的持续时间
        
        # --- 奖励函数参数 ---
        RTH: float = 0.10,             # 到达目标点的判定半径阈值 (Reach Threshold)
        CTH: float = 0.15,             # 碰撞判定阈值 (Collision Threshold)
        r_reach: float = 10.0,        # 到达目标的稀疏奖励
        r_collision: float = -10.0,   # 发生碰撞的惩罚
        p_r: float = 10,               # 势能奖励系数（靠近目标得分，远离扣分）
        r_o: float = -0.02,             # 时间步惩罚（鼓励尽快到达）
        
        # --- 目标设置 ---
        waypoints: list[tuple[float, float]] | None = None, # 任务路径点列表
        waypoint_rth: float = 0.10,    # 路径点打卡半径
        max_goal_distance: float = 8.0,# 状态归一化用的最大距离参考值
        
        # --- 系统配置 ---
        wait_timeout: float = 1.0,     # 等待 ROS 服务的超时时间
        
        # --- 动态障碍物配置 ---
        obstacle_mode: bool = False,   # 是否开启障碍物模式
        obstacle_models_root: str | None = None, # 障碍物 SDF 模型根目录
        debug_obstacles: bool = False, # 是否打印障碍物调试信息
        
        # --- 可视化配置 ---
        enable_viz: bool = True,       # 是否在 Rviz 中发布标记
        viz_frame: str = "odom",       # 可视化坐标系
        max_path_len: int = 3000,      # 轨迹标记最大保留长度
        render_mode: str | None = None,
        
    ):
        super().__init__()
        ensure_ros_init() # 初始化 ROS 节点

        # 成员变量保存
        self.scan_topic = scan_topic
        self.odom_topic = odom_topic
        self.cmd_vel_topic = cmd_vel_topic
        self.reset_world_service = reset_world_service
        self.reset_sim_service = reset_sim_service
        self.set_model_state_service = set_model_state_service
        self.robot_model_name = robot_model_name
        
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
        
        self.waypoints = waypoints
        if not self.waypoints:
            raise ValueError("waypoints must be provided and non-empty")
        self.waypoint_rth = waypoint_rth
        self.wp_idx = 0 # 当前追踪的路径点索引
        self.max_goal_distance = max_goal_distance
        self.wait_timeout = wait_timeout
        self.obstacle_mode = obstacle_mode
        
        # 确定障碍物模型路径（默认从 kfdqn_gazebo 包中获取）
        if obstacle_models_root is None:
            obstacle_models_root = os.path.join(
                rospkg.RosPack().get_path("kfdqn_gazebo"),
                "models",
            )
        self.obstacle_models_root = obstacle_models_root
        self.debug_obstacles = debug_obstacles
        
        self.enable_viz = enable_viz
        self.viz_frame = viz_frame
        self.max_path_len = max_path_len
        
        # ---------------------------------------------------------------------
        # 定义 Gym 空间
        # ---------------------------------------------------------------------
        # 离散动作空间：0, 1, 2
        self.action_space = spaces.Discrete(3)
        # 观测空间：90(雷达归一化) + 1(目标相对角度) + 1(距离归一化) + 1(上一步动作归一化)
        obs_low = np.array([0.0] * 90 + [-1.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0] * 90 + [1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ---------------------------------------------------------------------
        # 初始化 ROS 通讯
        # ---------------------------------------------------------------------
        self._cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        
        self._current_scan: Optional[LaserScan] = None
        self._current_odom: Optional[Odometry] = None

        # 异步订阅数据，更新类成员变量
        self._scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self._scan_cb, queue_size=1)
        self._odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)

        # 缓存 Gazebo 服务代理，用于重置和生成物体
        self._srv_set_state = None
        self._srv_spawn = None
        self._srv_delete = None
        if _HAS_MODEL_STATE:
            self._srv_set_state = rospy.ServiceProxy(self.set_model_state_service, SetModelState)
        if _HAS_MODEL_SPAWN:
            self._srv_spawn = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            self._srv_delete = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)

        # Rviz 标记发布者
        if self.enable_viz:
            self._waypoints_pub = rospy.Publisher("/kfdqn_viz/waypoints", MarkerArray, queue_size=1, latch=True)
            self._current_wp_pub = rospy.Publisher("/kfdqn_viz/current_wp", Marker, queue_size=1)
            self._trajectory_pub = rospy.Publisher("/kfdqn_viz/trajectory", Path, queue_size=1)

        self._np_random = np.random.default_rng()
        self.prev_action = 0.0
        self.prev_dis = 0.0
        self.step_count = 0
        
        # 初始化当前目标点
        if self.waypoints:
            self.goal = np.array(self.waypoints[0], dtype=np.float32)
        else:
            self.goal = np.array([0.0, 0.0], dtype=np.float32)
        
        self.path_msg: Optional[Path] = None
        self._sdf_cache: dict[str, str] = {}
        self._fixed_obstacles_spawned = False # 标记障碍物是否已在 Gazebo 中生成

    def _scan_cb(self, msg: LaserScan):
        """雷达回调函数"""
        self._current_scan = msg

    def _odom_cb(self, msg: Odometry):
        """里程计回调函数"""
        self._current_odom = msg

    def _load_model_sdf(self, model_dir: str) -> str:
        """从文件加载 SDF 内容并缓存，提高生成效率"""
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

    def _ensure_fixed_obstacles_spawned(self) -> None:
        """确保静态障碍物在 Gazebo 中存在，若不存在则调用服务生成"""
        if self._fixed_obstacles_spawned:
            return
        # 先尝试删除同名模型防止冲突
        self._delete_obstacle("box_a")
        self._delete_obstacle("box_b")
        if not self._spawn_model_from_dir("box_a", "box_a", x=0.0, y=0.0, z=0.0, yaw=0.0):
            raise RuntimeError("Failed to spawn fixed obstacle: box_a")
        if not self._spawn_model_from_dir("box_b", "box_b", x=0.0, y=0.0, z=0.0, yaw=0.0):
            raise RuntimeError("Failed to spawn fixed obstacle: box_b")
        self._fixed_obstacles_spawned = True

    def _call_reset(self) -> None:
        """重置仿真世界，尝试不同的重置策略"""
        try:
            rospy.wait_for_service(self.reset_world_service, timeout=self.wait_timeout)
            rospy.ServiceProxy(self.reset_world_service, Empty)()
            return
        except Exception:
            pass

        try:
            rospy.wait_for_service(self.reset_sim_service, timeout=self.wait_timeout)
            rospy.ServiceProxy(self.reset_sim_service, Empty)()
        except Exception:
            if getattr(self, "debug_obstacles", False):
                print(f"[reset] failed to call {self.reset_world_service} and {self.reset_sim_service}")

    def _call_set_model_state(self) -> None:
        """将机器人传送回初始位置并强制速度清零"""
        if (self.robot_model_name is None) or (not _HAS_MODEL_STATE):
            return

        try:
            rospy.wait_for_service(self.set_model_state_service, timeout=self.wait_timeout)
            srv = self._srv_set_state
            if srv is None:
                return
            state = ModelState()
            state.model_name = self.robot_model_name
            state.pose.position.x = float(self.init_x)
            state.pose.position.y = float(self.init_y)
            state.pose.position.z = 0.03 # 略微抬高防止与地面物理重叠
            yaw = float(self.init_yaw)
            state.pose.orientation.z = math.sin(yaw / 2.0)
            state.pose.orientation.w = math.cos(yaw / 2.0)
            # 强制清空残余动量
            state.twist.linear.x = state.twist.linear.y = state.twist.linear.z = 0.0
            state.twist.angular.x = state.twist.angular.y = state.twist.angular.z = 0.0
            srv(state)
        except Exception as exc:
            if getattr(self, "debug_obstacles", False):
                print(f"[set_model_state] failed: {exc}")

    def _spawn_model_from_dir(self, model_name, model_dir, x=0.0, y=0.0, z=0.0, yaw=0.0) -> bool:
        """通过目录名加载并生成模型"""
        sdf_xml = self._load_model_sdf(model_dir)
        return self._call_spawn_sdf_model(model_name, sdf_xml, x, y, z, yaw)

    def _delete_obstacle(self, name: str) -> bool:
        """从仿真中删除指定名称的模型"""
        if not _HAS_MODEL_SPAWN: return False
        try:
            rospy.wait_for_service("/gazebo/delete_model", timeout=self.wait_timeout)
            srv = self._srv_delete
            if srv is None: return False
            resp = srv(name)
            return bool(getattr(resp, "success", True))
        except Exception: return False

    def _call_spawn_sdf_model(self, model_name, sdf_xml, x, y, z, yaw) -> bool:
        """调用 Gazebo 物理引擎生成物体"""
        if not _HAS_MODEL_SPAWN: return False
        pose = Pose()
        pose.position.x = float(x); pose.position.y = float(y); pose.position.z = float(z)
        pose.orientation.z = math.sin(yaw / 2.0); pose.orientation.w = math.cos(yaw / 2.0)
        try:
            rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=self.wait_timeout)
            srv = self._srv_spawn
            if srv is None: return False
            resp = srv(model_name, sdf_xml, "", pose, "world")
            return bool(getattr(resp, "success", True))
        except Exception: return False

    def _publish_cmd(self, linear_x: float, angular_z: float) -> None:
        """发布 Twist 控制指令"""
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self._cmd_pub.publish(cmd)

    def _get_pose(self, odom: Odometry) -> Tuple[float, float, float]:
        """从里程计消息中解析出 X, Y 和偏航角"""
        pos = odom.pose.pose.position
        ori = odom.pose.pose.orientation
        yaw = _yaw_from_quat(ori.x, ori.y, ori.z, ori.w)
        return float(pos.x), float(pos.y), float(yaw)

    def current_goal(self) -> Tuple[float, float]:
        """获取当前正在追踪的路径点坐标"""
        return self.waypoints[self.wp_idx]

    def _lidar90(self, scan: LaserScan) -> np.ndarray:
        """将原始雷达射线（通常较多）重采样压缩为固定的 90 维数据点"""
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
            segment = ranges[start:end] if end > start else ranges[start:start+1]
            finite = segment[np.isfinite(segment)]
            bins[i] = float(np.min(finite)) if finite.size > 0 else self.max_lidar_range
        
        bins = np.clip(bins, 0.0, self.max_lidar_range)
        return bins.astype(np.float32)

    def _normalize_obs(self, lidar: np.ndarray, theta_d: float, dis: float, prev_action: float) -> np.ndarray:
        """将状态数据映射到神经网络喜欢的范围，如 [0, 1] 或 [-1, 1]"""
        lidar_norm = lidar / self.max_lidar_range
        theta_norm = theta_d / math.pi
        norm_dist = self.max_goal_distance
        dis_norm = np.clip(dis / norm_dist, 0.0, 1.0)
        return np.concatenate([lidar_norm, np.array([theta_norm, dis_norm, prev_action], dtype=np.float32)], axis=0)

    # -------------------------------------------------------------------------
    # 可视化标记发布逻辑
    # -------------------------------------------------------------------------
    def _publish_waypoints_markers(self) -> None:
        """在 Rviz 中渲染所有路径点（蓝色半透明球体）"""
        if not self.enable_viz: return
        points = self.waypoints if self.waypoints else []
        marker_array = MarkerArray()
        for idx, (gx, gy) in enumerate(points):
            marker = Marker()
            marker.header.frame_id = self.viz_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"; marker.id = idx; marker.type = Marker.SPHERE; marker.action = Marker.ADD
            marker.pose.position.x = gx; marker.pose.position.y = gy; marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0; marker.scale.x = marker.scale.y = marker.scale.z = 0.12
            marker.color.b = 1.0; marker.color.a = 0.6
            marker_array.markers.append(marker)
        self._waypoints_pub.publish(marker_array)

    def _publish_current_wp_marker(self) -> None:
        """高亮显示当前需要到达的目标（红色不透明球体）"""
        if not self.enable_viz: return
        gx, gy = self.current_goal()
        marker = Marker()
        marker.header.frame_id = self.viz_frame; marker.header.stamp = rospy.Time.now()
        marker.ns = "current_wp"; marker.id = 999; marker.type = Marker.SPHERE; marker.action = Marker.ADD
        marker.pose.position.x = gx; marker.pose.position.y = gy; marker.pose.position.z = 0.06
        marker.pose.orientation.w = 1.0; marker.scale.x = marker.scale.y = marker.scale.z = 0.18
        marker.color.r = 1.0; marker.color.a = 1.0
        self._current_wp_pub.publish(marker)

    # -------------------------------------------------------------------------
    # Gym Reset: 回合重置逻辑
    # -------------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境状态，准备开始新的一局训练"""
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            
        max_retries = 100 
        for _ in range(max_retries):
            self.wp_idx = 0 # 重置路径点索引到起点
            
            # 记录重置前的序列号，用于检测数据刷新
            odom_seq0 = self._current_odom.header.seq if self._current_odom else -1
            scan_seq0 = self._current_scan.header.seq if self._current_scan else -1
            self._current_scan = self._current_odom = None

            # self._call_reset()
            self._call_set_model_state()
            self._publish_cmd(0.0, 0.0)

            if self.obstacle_mode:
                self._ensure_fixed_obstacles_spawned()
            
            # 等待传感器接收到重置后的、且位置正确的数据
            start_wait = time.time()
            data_valid = False
            while True:
                scan, odom = self._current_scan, self._current_odom
                if scan and odom and scan.header.seq > scan_seq0 and odom.header.seq > odom_seq0:
                    x, y, _ = self._get_pose(odom)
                    # 验证机器人是否真的出现在初始坐标附近
                    if (x - self.init_x) ** 2 + (y - self.init_y) ** 2 < (0.15 ** 2):
                        data_valid = True; break
                if time.time() - start_wait > 2.0: break 
                time.sleep(0.005)
            
            if not data_valid: continue

            # [核心修复] 检查出生点是否安全（防止重置后立马撞墙）
            lidar_check = self._lidar90(self._current_scan)
            if np.min(lidar_check) < self.CTH + 0.05: continue
            break

        # 重置轨迹可视化
        if self.enable_viz:
            self.path_msg = Path()
            self.path_msg.header.frame_id = self.viz_frame
            self._trajectory_pub.publish(self.path_msg) 

        self.step_count = 0
        self.prev_action = 0.0
        gx, gy = self.current_goal()
        self.goal = np.array([gx, gy], dtype=np.float32)

        if self.enable_viz:
            self._publish_waypoints_markers()
            self._publish_current_wp_marker()

        # 计算初始状态信息
        lidar = self._lidar90(self._current_scan)
        x, y, yaw = self._get_pose(self._current_odom)
        dx, dy = gx - x, gy - y
        dis = float(math.hypot(dx, dy))
        theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)
        self.prev_dis = dis

        obs = self._normalize_obs(lidar, theta_d, dis, self.prev_action)
        info = {"min_lidar": float(np.min(lidar)), "theta_d": float(theta_d), "dis": float(dis)}
        return obs.astype(np.float32), info

    # -------------------------------------------------------------------------
    # Gym Step: 执行动作并更新环境
    # -------------------------------------------------------------------------
    def step(self, action: int):
        """核心训练步：发布动作 -> 等待 -> 计算奖励 -> 返回观测"""
        assert self.action_space.contains(action)
        self.step_count += 1

        # 根据动作选择对应的速度指令
        if action == self.ACTION_LEFT: lin, ang = self.turn_v, self.turn_omega
        elif action == self.ACTION_RIGHT: lin, ang = self.turn_v, -self.turn_omega
        else: lin, ang = self.forward_v, 0.0

        scan_seq0 = self._current_scan.header.seq if self._current_scan else -1

        # 持续发布指定频率的指令，确保动作执行平稳
        start_time = rospy.get_time()
        end_time = start_time + self.action_duration
        rate = rospy.Rate(self.publish_hz)
        while rospy.get_time() < end_time:
            if rospy.is_shutdown(): break
            try:
                self._publish_cmd(lin, ang)
                rate.sleep()
            except rospy.ROSTimeMovedBackwardsException: break
        
        # 等待动作执行后的最新一帧传感器数据
        t0, timeout = rospy.get_time(), 0.5 
        rate_wait = rospy.Rate(100)
        while True:
            scan, odom = self._current_scan, self._current_odom
            if scan and odom and scan.header.seq > scan_seq0: break
            if rospy.get_time() - t0 > timeout: break
            rate_wait.sleep()
            
        if scan is None or odom is None: raise RuntimeError("Data loss during step.")

        # 可视化实时运行轨迹
        if self.enable_viz:
            if self.path_msg is None: self.path_msg = Path(); self.path_msg.header.frame_id = self.viz_frame
            pose = PoseStamped(); pose.header.frame_id = self.viz_frame; pose.header.stamp = rospy.Time.now()
            pose.pose = odom.pose.pose; self.path_msg.poses.append(pose)
            if len(self.path_msg.poses) > self.max_path_len: self.path_msg.poses = self.path_msg.poses[-self.max_path_len :]
            self._trajectory_pub.publish(self.path_msg)

        # 状态提取
        lidar = self._lidar90(scan)
        min_lidar = float(np.min(lidar))
        x, y, yaw = self._get_pose(odom)
        gx, gy = self.current_goal()
        dx, dy = gx - x, gy - y
        dis = float(math.hypot(dx, dy))
        theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)

        terminated = truncated = False
        info = {"min_lidar": float(min_lidar), "theta_d": float(theta_d), "dis": float(dis)}

        # --- 奖励计算核心逻辑 ---
        if min_lidar < self.CTH:
            # 1. 发生碰撞
            reward = self.r_collision
            terminated = True
            info["is_collision"] = True
        else:
            # 2. 正常行驶：势能奖励 + 每步代价
            reward = (self.prev_dis - dis) * self.p_r + self.r_o
            
            # 到达当前路径点阈值内
            if dis < self.waypoint_rth:
                if self.wp_idx < len(self.waypoints) - 1:
                    # 打卡中间路径点，切换到下一个
                    self.wp_idx += 1
                    new_gx, new_gy = self.current_goal()
                    dx, dy = new_gx - x, new_gy - y
                    dis = float(math.hypot(dx, dy))
                    theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)
                    self.prev_dis = dis
                    reward += self.r_reach
                    info["waypoint_reached"] = True; info["wp_idx"] = self.wp_idx
                    if self.enable_viz: self._publish_current_wp_marker()
                else:
                    # 到达终点，回合结束
                    reward = self.r_reach
                    terminated = True
                    info["is_success"] = True

        # 步数上限检查
        if self.step_count >= self.max_steps: truncated = True

        # 更新历史信息
        self.prev_action = float(action) / 2.0
        if not terminated: self.prev_dis = dis
        
        # 任务结束时停止机器人
        if terminated or truncated: self._publish_cmd(0.0, 0.0)

        obs = self._normalize_obs(lidar, theta_d, dis, self.prev_action)
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def close(self) -> None:
        """关闭环境并发布停止指令"""
        self._publish_cmd(0.0, 0.0)

# --- 环境注册：通过 gym.make(id) 调用时，使用不同的 kwargs 构造特定的场景 ---

if "GoalReach-v0" not in registry:
    register(
        id="GoalReach-v0",
        entry_point="envs_ros.env_eval:ROSGazeboMobileRobotEnv",
        kwargs={
            "obstacle_mode": False,
            "robot_model_name": "turtlebot3_burger",
            "waypoints": [(0.7, 0.5), (1.6, 0.5), (1.3, -0.4), (0.8, 0.0), (0.8, -0.5)],
            "waypoint_rth": 0.20,
            "max_steps": 500,
        },
    )

if "ObstacleAvoid-v0" not in registry:
    register(
        id="ObstacleAvoid-v0",
        entry_point="envs_ros.env_eval:ROSGazeboMobileRobotEnv",
        kwargs={
            "obstacle_mode": True,
            "robot_model_name": "turtlebot3_burger",
            "waypoints": [(1.4, 0.9), (-0.15, 0.1)],
            "waypoint_rth": 0.20,
            "max_steps": 1000,
        },
    )