from __future__ import annotations

import math
import time
from typing import Optional, Tuple

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

# --- 检查 Gazebo 服务依赖：判断当前环境下 Gazebo 相关的消息和服务是否可用 ---
try:
    from gazebo_msgs.srv import SetModelState
    from gazebo_msgs.msg import ModelState

    _HAS_MODEL_STATE = True
except Exception:
    _HAS_MODEL_STATE = False

# 全局变量：用于跟踪 ROS 节点是否已经初始化，防止多重初始化引发异常
_ROS_INITIALIZED = False


def ensure_ros_init(node_name: str = "gym_mobile_robot_env") -> None:
    """
    静态工具函数：确保在一个进程中 ROS 节点只初始化一次。
    如果检测到 ROS 已由外部或先前调用初始化，则直接返回。
    """
    global _ROS_INITIALIZED
    if _ROS_INITIALIZED or rospy.core.is_initialized():
        return
    rospy.init_node(node_name, anonymous=True, disable_signals=True)
    _ROS_INITIALIZED = True


def _wrap_angle(angle: float) -> float:
    """
    数学辅助函数：将任意角度值（弧度）归一化到 [-pi, pi] 之间。
    这在计算机器人相对于目标的航向角偏差时非常关键。
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """
    坐标转换：将 ROS 常见的四元数朝向信息转换为单轴偏航角 (Yaw)。
    适用于 2D 平面导航任务，提取机器人在 XY 平面上的旋转角度。
    """
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - (2.0 * (y * y + z * z))
    return math.atan2(siny, cosy)


class ROSGazeboMobileRobotTrainEnv(gym.Env):
    """
    ROS Gazebo 移动机器人训练环境 (封装为 Gymnasium 标准接口)
    
    设计理念：
    1. 针对强化学习训练优化：支持随机出生点和随机目标采样，提升泛化性。
    2. 高维观察空间：90维降采样雷达数据提供丰富的环境感知。
    3. 同步控制逻辑：确保每步动作执行后，状态观测值对应物理引擎的最新时刻。
    """

    metadata = {"render_modes": []}

    # 离散动作空间索引映射
    ACTION_LEFT = 0     # 左转并前进
    ACTION_RIGHT = 1    # 右转并前进
    ACTION_FORWARD = 2  # 纯直线前进

    def __init__(
        self,
        *,
        # --- ROS 话题与服务配置 ---
        scan_topic: str = "/scan",           # 激光雷达订阅话题
        odom_topic: str = "/odom",           # 里程计订阅话题
        cmd_vel_topic: str = "/cmd_vel",     # 控制指令发布话题
        reset_world_service: str = "/gazebo/reset_world",     # 重置物理环境服务
        reset_sim_service: str = "/gazebo/reset_simulation",  # 重置仿真时间服务
        set_model_state_service: str = "/gazebo/set_model_state", # 强制设置模型位姿服务
        robot_model_name: str | None = None, # Gazebo 中机器人的模型名称标识

        # --- 训练参数与物理特性 ---
        max_steps: int = 100,                # 回合最大步数限制
        max_lidar_range: float = 3.5,        # 雷达有效截断距离
        forward_v: float = 0.2,             # 直行线速度 (m/s)
        turn_v: float = 0.2,                # 转向时的线速度 (m/s)
        turn_omega: float = math.pi/2,             # 转向时的角速度 (rad/s)
        publish_hz: float = 30.0,            # 指令发布频率 (Hz)
        action_duration: float = 0.1,        # 单步动作物理执行持续时间 (s)

        # --- 奖励函数参数 (Reward Shaping) ---
        RTH: float = 0.20,                   # 到达目标的物理距离半径
        CTH: float = 0.15,                   # 碰撞触发的最小避障安全距离
        r_reach: float = 10.0,              # 成功到达目标点的奖励 (Positive Reward)
        r_collision: float = -5.0,         # 发生碰撞后的惩罚 (Negative Reward)
        p_r: float = 10,                     # 势能奖励系数 (基于距离目标的接近程度)
        r_o: float = -0.02,                   # 时间步生存惩罚 (鼓励最短路径到达)

        # --- 环境约束与阈值 ---
        waypoint_rth: float = 0.20,          # 航点到达判定阈值
        max_goal_distance: float = 5.5,      # 观察空间中距离归一化的基准值

        # --- 系统稳定性配置 ---
        wait_timeout: float = 1.0,           # 等待 ROS 服务响应的超时时长
        obstacle_mode: bool = False,         # 是否处于复杂避障训练模式
        debug_obstacles: bool = False,       # 是否打印障碍物处理相关的调试信息

        # --- 随机重置配置 ---
        map_xy_limit: float = 2.0,           # 采样区域的 XY 坐标绝对值边界
        wall_margin: float = 0.35,           # 采样出生点时离墙壁的保护边距
        goal_d_min: float = 0.5,             # 目标点离机器人出生的最小允许距离
        goal_d_max: float = 3,               # 目标点离机器人出生的最大允许距离
        safety_margin: float = 0.05,         # 重置时额外的碰撞检测安全余量
        max_reset_retries: int = 150,        # 重置时采样合法点的最大重试次数
        continue_on_success: bool = False,   # 成功到达后是否继续累积回合（不重置机器人）

        # --- Rviz 可视化配置 ---
        enable_viz: bool = True,             # 是否发布 Marker 和 Path 供可视化分析
        viz_frame: str = "odom",             # 可视化坐标系参考帧
        max_path_len: int = 1000,            # 可视化轨迹点数上限
        render_mode: str | None = None,      # 符合 Gym 接口要求的渲染模式占位
    ):
        super().__init__()
        ensure_ros_init() # 确保本进程内 ROS 节点已运行

        # 成员变量初始化：保存所有超参数
        self.scan_topic = scan_topic
        self.odom_topic = odom_topic
        self.cmd_vel_topic = cmd_vel_topic
        self.reset_world_service = reset_world_service
        self.reset_sim_service = reset_sim_service
        self.set_model_state_service = set_model_state_service
        self.robot_model_name = robot_model_name
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
        self.waypoint_rth = waypoint_rth
        self.max_goal_distance = max_goal_distance
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

        # --- 定义 Gymnasium 标准空间 (符合 KFDQN 论文设定) ---
        self.action_space = spaces.Discrete(3) # 离散动作 [0, 1, 2]
        # 观察空间构成: 90(雷达) + 1(目标角度) + 1(目标距离) + 1(上一步动作) = 93维特征
        obs_low = np.array([0.0] * 90 + [-1.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0] * 90 + [1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # --- 初始化 ROS 通信组件 ---
        self._cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self._current_scan: Optional[LaserScan] = None
        self._current_odom: Optional[Odometry] = None
        
        # 订阅激光雷达和里程计数据，实时刷新成员变量
        self._scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self._scan_cb, queue_size=1)
        self._odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)

        # 初始化 Gazebo 位姿管理服务
        self._srv_set_state = None
        if _HAS_MODEL_STATE:
            self._srv_set_state = rospy.ServiceProxy(self.set_model_state_service, SetModelState)

        # 可视化发布器初始化
        if self.enable_viz:
            self._current_wp_pub  = rospy.Publisher("/kfdqn_viz/current_wp", Marker, queue_size=1)
            self._trajectory_pub = rospy.Publisher("/kfdqn_viz/trajectory", Path, queue_size=1)

        # 内部状态变量记录
        self._np_random = np.random.default_rng()
        self.prev_action = 0.0 # 用于状态输入
        self.prev_dis = 0.0    # 用于计算位能奖励
        self.step_count = 0    # 回合步数统计

        self.init_x = 0.0      # 采样确定的起始 X
        self.init_y = 0.0      # 采样确定的起始 Y
        self.init_yaw = 0.0    # 采样确定的起始偏航角
        self.goal = np.array([0.0, 0.0], dtype=np.float32) # 当前目标点坐标

        self.path_msg: Optional[Path] = None # 用于 Rviz 轨迹显示的路径消息

        self.obstacles = [
            (-0.6, -0.6, 0.35), # 左下
            (-0.6,  0.6, 0.35), # 左上
            ( 0.6, -0.6, 0.35), # 右下
            ( 0.6,  0.6, 0.35), # 右上
            ( 1.7,  0.0, 0.35), # 右边界中心
            (-1.7,  0.0, 0.35), # 左边界中心
            ( 0.0,  1.7, 0.35), # 上边界中心
            ( 0.0, -1.7, 0.35), # 下边界中心
        ]

    def _scan_cb(self, msg: LaserScan):
        """激光雷达订阅回调：缓存最新的一帧传感器数据对象"""
        self._current_scan = msg

    def _odom_cb(self, msg: Odometry):
        """里程计订阅回调：缓存最新的机器人实时位姿和速度数据对象"""
        self._current_odom = msg

    def _call_reset(self) -> None:
        """
        内部逻辑：调用 Gazebo 重置服务。
        优先重置世界物理，失败则尝试重置整个仿真逻辑。
        """
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
        """
        强制干预：通过 Gazebo 服务将机器人瞬间平移到采样的 init 坐标。
        包含重置朝向、清空残余线速度和角速度，确保回合开始时的物理独立性。
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
            state.pose.position.x = float(self.init_x)
            state.pose.position.y = float(self.init_y)
            state.pose.position.z = 0.03 # 略高于地面防止嵌入
            yaw = float(self.init_yaw)
            state.pose.orientation.z = math.sin(yaw / 2.0)
            state.pose.orientation.w = math.cos(yaw / 2.0)
            # 清空速度矢量，防止继承上一个回合的惯性
            state.twist.linear.x = state.twist.linear.y = state.twist.linear.z = 0.0
            state.twist.angular.x = state.twist.angular.y = state.twist.angular.z = 0.0
            srv(state)
        except Exception as exc:
            if getattr(self, "debug_obstacles", False):
                print(f"[set_model_state] failed: {exc}")

    def _publish_cmd(self, linear_x: float, angular_z: float) -> None:
        """底层执行：将线速度和角速度发布至 cmd_vel 驱动机器人"""
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self._cmd_pub.publish(cmd)

    def _get_pose(self, odom: Odometry) -> Tuple[float, float, float]:
        """数据解析：从 Odometry 消息中提取 2D 坐标 (X, Y) 和 偏航角 (Yaw)"""
        pos = odom.pose.pose.position
        ori = odom.pose.pose.orientation
        yaw = _yaw_from_quat(ori.x, ori.y, ori.z, ori.w)
        return float(pos.x), float(pos.y), float(yaw)

    def _lidar90(self, scan: LaserScan) -> np.ndarray:
        """
        雷达数据预处理 (等间隔采样模式)：
        每隔 4° 采样一个数据点，总共采样 90 个点 (360 / 4 = 90)。
        """
        if scan.range_max > 0.0:
            self.max_lidar_range = float(scan.range_max)

        ranges = np.array(list(scan.ranges), dtype=np.float32)
        n = ranges.size

        if n == 0:
            return np.full(90, self.max_lidar_range, dtype=np.float32)

        # 1. 预处理：处理原始数据中的 inf 和 nan
        # 采样模式下，如果刚好采到 inf，必须将其视为最大距离，否则网络会报错
        ranges[~np.isfinite(ranges)] = self.max_lidar_range
        ranges = np.clip(ranges, 0.0, self.max_lidar_range)

        # 2. 计算采样索引
        # 假设雷达是 360 度覆盖：
        # - 如果 n=360 (分辨率1度)，则 stride=4，取索引 0, 4, 8...
        # - 如果 n=720 (分辨率0.5度)，则 stride=8，取索引 0, 8, 16...
        # 这里的 n / 90.0 可以自动适应不同分辨率的雷达
        indices = (np.arange(90) * (n / 90.0)).astype(int)

        # 3. 直接通过索引采样
        bins = ranges[indices]

        return bins.astype(np.float32)

    def _normalize_obs(self, lidar: np.ndarray, theta_d: float, dis: float, prev_action: float) -> np.ndarray:
        """
        观察向量归一化：
        1. 雷达值除以最大量程 -> [0, 1]
        2. 角度除以 Pi -> [-1, 1]
        3. 距离根据基准缩放并截断 -> [0, 1]
        最终拼接成 93 维输入向量。
        """
        lidar_norm = lidar / self.max_lidar_range
        theta_norm = theta_d / math.pi
        norm_dist = self.max_goal_distance
        dis_norm = np.clip(dis / norm_dist, 0.0, 1.0)
        return np.concatenate(
            [lidar_norm, np.array([theta_norm, dis_norm, prev_action], dtype=np.float32)], axis=0
        )

    def _sample_uniform_xy(self) -> Tuple[float, float]:
        """在地图安全边界内随机采样一个均匀分布的坐标点"""
        lim = self.map_xy_limit - self.wall_margin
        x = float(self._np_random.uniform(-lim, lim))
        y = float(self._np_random.uniform(-lim, lim))
        return x, y

    def _sample_robot_pose(self) -> Tuple[float, float, float]:
        """随机采样机器人的起始位姿和初始随机朝向"""
        x, y = self._sample_uniform_xy()
        yaw = float(self._np_random.uniform(-math.pi, math.pi))
        # return x, y, yaw
        return 0, 0, 0

    def _sample_goal(self, robot_x: float, robot_y: float) -> Tuple[float, float]:
        """
        随机目标采样 (带约束)：
        1. 距离机器人 min ~ max 范围。
        2. [新增] 不生成在已知障碍物内部。
        """
        lim = self.map_xy_limit - self.wall_margin
        
        for _ in range(1000): # 增加尝试次数
            gx = float(self._np_random.uniform(-lim, lim))
            gy = float(self._np_random.uniform(-lim, lim))
            
            # 1. 距离检查
            d = float(math.hypot(gx - robot_x, gy - robot_y))
            if not (self.goal_d_min <= d <= self.goal_d_max):
                continue
                
            # 2. [新增] 障碍物碰撞检查
            is_valid = True
            for ox, oy, r in self.obstacles:
                # 计算目标点到障碍物中心的距离
                dist_to_obs = math.hypot(gx - ox, gy - oy)
                if dist_to_obs < r: # 如果落在障碍物半径内
                    is_valid = False
                    break
            
            if is_valid:
                return gx, gy
                
        # 如果实在找不到，回退到原来的逻辑（或者抛出更详细的错误）
        print(f"Warning: Could not find valid goal away from obstacles for robot at ({robot_x:.2f}, {robot_y:.2f})")
        # 即使失败也返回一个随机点，防止程序崩溃，但在训练中可能会有些“坏”数据
        return gx, gy

    def _publish_current_wp_marker(self) -> None:
        """在 Rviz 中发布一个红色球体，直观显示机器人当前需要追逐的目标位置"""
        if not self.enable_viz:
            return
        gx, gy = float(self.goal[0]), float(self.goal[1])

        marker = Marker()
        marker.header.frame_id = self.viz_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "current_wp"
        marker.id = 999
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = gx
        marker.pose.position.y = gy
        marker.pose.position.z = 0.06
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.a = 1.0

        self._current_wp_pub.publish(marker)

    # -------------------------------------------------------------------------
    # Gym Reset: 回合初始化核心
    # -------------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        环境重置逻辑：
        1. 物理重置 Gazebo 状态。
        2. 采样随机安全位置。
        3. 采样随机目标。
        4. 等待传感器反馈确认状态同步成功（通过 header.seq 检查）。
        """
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # 试错采样：确保重置后机器人不处于碰撞状态
        for _ in range(self.max_reset_retries):
            rx, ry, ryaw = self._sample_robot_pose()

            # 记录重置前的数据时间戳
            odom_seq0 = self._current_odom.header.seq if self._current_odom else -1
            scan_seq0 = self._current_scan.header.seq if self._current_scan else -1
            self._current_scan = self._current_odom = None # 显式清空以等待新数据

            # self._call_reset() # 全局重置

            self.init_x, self.init_y, self.init_yaw = rx, ry, ryaw
            self._call_set_model_state() # 强制位姿重置
            self._publish_cmd(0.0, 0.0)  # 强制静止

            # 阻塞等待逻辑：确保仿真器已经完成了位姿切换并发布了新的传感器包
            start_wait = time.time()
            data_valid = False
            while True:
                scan, odom = self._current_scan, self._current_odom
                if scan and odom and scan.header.seq > scan_seq0 and odom.header.seq > odom_seq0:
                    x, y, _ = self._get_pose(odom)
                    # 验证 Gazebo 服务是否真正将位置切换到了采样点
                    if (x - rx) ** 2 + (y - ry) ** 2 < (0.15**2):
                        data_valid = True
                        break
                if time.time() - start_wait > 2.0:
                    break
                time.sleep(0.005)

            if not data_valid:
                continue # 数据不同步，重新执行一次重置逻辑

            # 出生点安全性检查：雷达若检测到机器人出生在障碍物里，则重新采样
            lidar_check = self._lidar90(self._current_scan)
            min_lidar = float(np.min(lidar_check))
            if min_lidar < self.CTH + self.safety_margin:
                continue

            # 确定合法目标并发布可视化
            gx, gy = self._sample_goal(rx, ry)
            self.goal = np.array([gx, gy], dtype=np.float32)
            if self.enable_viz:
                self._publish_current_wp_marker()
            break
        else:
            raise RuntimeError("Failed to reset env with a safe spawn.")

        # 初始化可视化轨迹对象
        if self.enable_viz:
            self.path_msg = Path()
            self.path_msg.header.frame_id = self.viz_frame
            self.path_msg.header.stamp = rospy.Time.now()
            self._trajectory_pub.publish(self.path_msg)


        self.step_count = 0
        self.prev_action = 0.0

        # 计算并返回初始状态观测
        lidar = self._lidar90(self._current_scan)
        x, y, yaw = self._get_pose(self._current_odom)
        dx, dy = float(self.goal[0]) - x, float(self.goal[1]) - y
        dis = float(math.hypot(dx, dy))
        theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)
        self.prev_dis = dis

        obs = self._normalize_obs(lidar, theta_d, dis, self.prev_action)
        info = {
            "min_lidar": float(np.min(lidar)),
            "theta_d": float(theta_d),
            "dis": float(dis),
            "goal": (float(self.goal[0]), float(self.goal[1])),
            "robot_pose": (float(x), float(y), float(yaw)),
        }
        return obs.astype(np.float32), info

    # -------------------------------------------------------------------------
    # Gym Step: 状态步进核心
    # -------------------------------------------------------------------------
    def step(self, action: int):
        """
        步进逻辑：
        1. 发布动作指令。
        2. 维持动作 action_duration 时长。
        3. 等待雷达数据刷新。
        4. 根据碰撞、距离缩减、生存惩罚计算奖励。
        """
        assert self.action_space.contains(action)
        self.step_count += 1

        # 映射离散动作到线速度/角速度
        if action == self.ACTION_LEFT:
            lin, ang = self.turn_v, self.turn_omega
        elif action == self.ACTION_RIGHT:
            lin, ang = self.turn_v, -self.turn_omega
        else:
            lin, ang = self.forward_v, 0.0

        scan_seq0 = self._current_scan.header.seq if self._current_scan else -1

        # 在 action_duration 窗口内持续发布控制频率指令
        start_time = rospy.get_time()
        end_time = start_time + self.action_duration
        rate = rospy.Rate(self.publish_hz)
        while rospy.get_time() < end_time:
            if rospy.is_shutdown():
                break
            try:
                self._publish_cmd(lin, ang)
                rate.sleep()
            except rospy.ROSTimeMovedBackwardsException:
                break

        # 指令发布完毕，等待物理仿真生效产生新一帧雷达观测
        t0, timeout = rospy.get_time(), 0.5
        rate_wait = rospy.Rate(100)
        while True:
            scan, odom = self._current_scan, self._current_odom
            if scan and odom and scan.header.seq > scan_seq0:
                break
            if rospy.get_time() - t0 > timeout:
                break
            rate_wait.sleep()

        if scan is None or odom is None:
            raise RuntimeError("Data loss during step.")

        # 轨迹可视化点入队
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
            self.path_msg.header.stamp = rospy.Time.now()
            self._trajectory_pub.publish(self.path_msg)

        # 解析新状态特征
        lidar = self._lidar90(scan)
        min_lidar = float(np.min(lidar))
        x, y, yaw = self._get_pose(odom)
        gx, gy = float(self.goal[0]), float(self.goal[1])
        dx, dy = gx - x, gy - y
        dis = float(math.hypot(dx, dy))
        theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)

        terminated = truncated = False
        info = {
            "min_lidar": float(min_lidar),
            "theta_d": float(theta_d),
            "dis": float(dis),
            "goal": (gx, gy),
        }

        # --- 奖励计算核心逻辑 (与 env_eval 逻辑完全对齐) ---
        if min_lidar < self.CTH:
            # 碰撞情况：给予显著负反馈并直接终止本回合
            reward = self.r_collision
            terminated = True
            info["is_collision"] = True
        else:
            # 正常导航：奖励 = 距离缩减增量奖励 + 固定的时间消耗惩罚
            reward = (self.prev_dis - dis) * self.p_r + self.r_o
            
            # 到达判定条件
            if dis < self.waypoint_rth:
                if self.continue_on_success:
                    # 连续导航模式：不重置机器人位姿，仅实时采样新目标点
                    info["is_success"] = True
                    reward = self.r_reach
                    new_gx, new_gy = self._sample_goal(x, y)
                    self.goal = np.array([new_gx, new_gy], dtype=np.float32)
                    if self.enable_viz:
                        self._publish_current_wp_marker()
                    # 更新增量奖励计算的基础值
                    dx, dy = new_gx - x, new_gy - y
                    dis = float(math.hypot(dx, dy))
                    theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)
                    self.prev_dis = dis
                    # 将新目标信息反馈给 Agent
                    info["goal_resampled"] = True
                    info["goal"] = (float(new_gx), float(new_gy))
                    info["dis"] = float(dis)
                    info["theta_d"] = float(theta_d)
                else:
                    # 单点模式：发放成功奖励并终止回合
                    reward = self.r_reach
                    terminated = True
                    info["is_success"] = True

        # 回合超时判定
        if self.step_count >= self.max_steps:
            truncated = True

        # 状态更新：记录本次动作和距离
        self.prev_action = float(action) / 2.0 # 动作离散值缩放
        if not terminated:
            self.prev_dis = dis

        # 若回合因各种原因结束，确保机器人立刻停止运动
        if terminated or truncated:
            self._publish_cmd(0.0, 0.0)

        obs = self._normalize_obs(lidar, theta_d, dis, self.prev_action)
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def close(self) -> None:
        """关闭环境时调用的清理函数：停止物理驱动"""
        self._publish_cmd(0.0, 0.0)


# -----------------------------------------------------------------------------
# 环境全局注册 (允许使用 gym.make 调用)
# -----------------------------------------------------------------------------
if "GoalReachTrain-v0" not in registry:
    register(
        id="GoalReachTrain-v0",
        entry_point="envs_ros.env_train:ROSGazeboMobileRobotTrainEnv",
        kwargs={
            "obstacle_mode": False,
            "robot_model_name": "turtlebot3_burger",
            "max_steps": 100,
        },
    )

if "ObstacleAvoidTrain-v0" not in registry:
    register(
        id="ObstacleAvoidTrain-v0",
        entry_point="envs_ros.env_train:ROSGazeboMobileRobotTrainEnv",
        kwargs={
            "obstacle_mode": True,
            "robot_model_name": "turtlebot3_burger",
            "max_steps": 100,
        },
    )