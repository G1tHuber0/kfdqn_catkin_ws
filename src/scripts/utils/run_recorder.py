# utils/run_recorder.py
from __future__ import annotations

import json
import os
import sys
import platform
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    from gymnasium import spaces as gym_spaces
except Exception:
    gym_spaces = None

def _to_jsonable(obj: Any) -> Any:
    """Convert common Python/NumPy/PyTorch objects into JSON-serializable forms."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # torch.device / dtype
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)

    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # numpy arrays / torch tensors: store metadata only (avoid huge JSON)
    if isinstance(obj, np.ndarray):
        return {"_type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if torch.is_tensor(obj):
        return {
            "_type": "tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }

    # containers
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    # fallback
    return str(obj)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _extract_optimizer_lrs(opt: Any) -> Optional[Dict[str, Any]]:
    """Return optimizer LR info if possible."""
    if opt is None:
        return None
    try:
        groups = []
        for i, g in enumerate(opt.param_groups):
            groups.append({"group": i, "lr": g.get("lr", None)})
        return {"type": opt.__class__.__name__, "param_groups": groups}
    except Exception:
        return {"type": opt.__class__.__name__}


def _extract_agent_info(agent: Any) -> Dict[str, Any]:
    """Best-effort extraction; works across DQN variants and KFDQN."""
    info: Dict[str, Any] = {"agent_class": agent.__class__.__name__}

    # common
    for k in ["epsilon", "gamma", "device"]:
        if hasattr(agent, k):
            info[k] = _safe_getattr(agent, k)

    # KFDQN-specific (record if present)
    for k in ["h1", "h2", "use_hybrid_learning"]:
        if hasattr(agent, k):
            info[k] = _safe_getattr(agent, k)

    # Some implementations keep m/n inside agent or update_parameters; record if present.
    for k in ["m", "n"]:
        if hasattr(agent, k):
            info[k] = _safe_getattr(agent, k)

    # Optimizers (names may vary; record if present)
    for opt_name in ["q_optimizer", "fuzzy_optimizer", "optimizer"]:
        if hasattr(agent, opt_name):
            info[opt_name] = _extract_optimizer_lrs(_safe_getattr(agent, opt_name))

    return info


def _space_min_max(values: Any) -> Dict[str, Optional[float]]:
    if values is None:
        return {"min": None, "max": None}
    try:
        arr = np.array(values, dtype=np.float64)
        if arr.size == 0:
            return {"min": None, "max": None}
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if not np.isfinite(min_val):
            min_val = None
        if not np.isfinite(max_val):
            max_val = None
        return {"min": min_val, "max": max_val}
    except Exception:
        return {"min": None, "max": None}


def _extract_space_info(space: Any) -> Dict[str, Any]:
    if space is None:
        return {}
    info: Dict[str, Any] = {"type": space.__class__.__name__}

    is_discrete = bool(gym_spaces) and isinstance(space, gym_spaces.Discrete)
    is_box = bool(gym_spaces) and isinstance(space, gym_spaces.Box)

    if not gym_spaces:
        is_discrete = space.__class__.__name__ == "Discrete"
        is_box = space.__class__.__name__ == "Box"

    if is_discrete:
        info["type"] = "Discrete"
        info["n"] = _safe_getattr(space, "n", None)
        if hasattr(space, "start"):
            info["start"] = _safe_getattr(space, "start", None)
        return info

    if is_box:
        info["type"] = "Box"
        info["shape"] = list(_safe_getattr(space, "shape", ()))
        info["dtype"] = str(_safe_getattr(space, "dtype", None))
        low_stats = _space_min_max(_safe_getattr(space, "low", None))
        high_stats = _space_min_max(_safe_getattr(space, "high", None))
        info["low_min"] = low_stats["min"]
        info["low_max"] = low_stats["max"]
        info["high_min"] = high_stats["min"]
        info["high_max"] = high_stats["max"]
        return info

    info["repr"] = str(space)
    return info


def _extract_env_info(env: Any) -> Dict[str, Any]:
    if env is None:
        return {}

    env_unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    env_class = env_unwrapped.__class__
    info: Dict[str, Any] = {
        "env_class": f"{env_class.__module__}.{env_class.__name__}",
    }

    spec = _safe_getattr(env, "spec", None)
    spec_id = _safe_getattr(spec, "id", None)
    if spec_id is not None:
        info["spec_id"] = spec_id

    action_space = _extract_space_info(_safe_getattr(env_unwrapped, "action_space", None))
    if action_space:
        info["action_space"] = action_space

    observation_space = _extract_space_info(_safe_getattr(env_unwrapped, "observation_space", None))
    if observation_space:
        info["observation_space"] = observation_space

    if action_space.get("type") == "Discrete":
        info["action_semantics"] = {"0": "left", "1": "right", "2": "forward"}
        action_constants = {}
        for name in ["ACTION_LEFT", "ACTION_RIGHT", "ACTION_FORWARD"]:
            if hasattr(env_unwrapped, name):
                action_constants[name] = _safe_getattr(env_unwrapped, name)
        if action_constants:
            info["action_constants"] = action_constants

    control = {
        "forward_v": _safe_getattr(env_unwrapped, "forward_v", None),
        "turn_v": _safe_getattr(env_unwrapped, "turn_v", None),
        "turn_omega": _safe_getattr(env_unwrapped, "turn_omega", None),
        "publish_hz": _safe_getattr(env_unwrapped, "publish_hz", None),
        "action_duration": _safe_getattr(env_unwrapped, "action_duration", None),
    }
    publish_hz = control.get("publish_hz")
    action_duration = control.get("action_duration")
    if publish_hz is not None and action_duration is not None:
        try:
            control["cmd_publishes_per_step"] = int(round(float(publish_hz) * float(action_duration)))
        except Exception:
            pass
    if any(value is not None for value in control.values()):
        info["control"] = control

    sensors = {
        "max_lidar_range": _safe_getattr(env_unwrapped, "max_lidar_range", None),
        "obs_comp": "90 lidar_norm + theta_norm + dis_norm + prev_action => 93 dims",
    }
    info["sensors"] = sensors

    episode = {
        "max_steps": _safe_getattr(env_unwrapped, "max_steps", None),
        "RTH": _safe_getattr(env_unwrapped, "RTH", None),
        "CTH": _safe_getattr(env_unwrapped, "CTH", None),
        "waypoint_rth": _safe_getattr(env_unwrapped, "waypoint_rth", None),
        "r_reach": _safe_getattr(env_unwrapped, "r_reach", None),
        "r_collision": _safe_getattr(env_unwrapped, "r_collision", None),
        "p_r": _safe_getattr(env_unwrapped, "p_r", None),
        "r_o": _safe_getattr(env_unwrapped, "r_o", None),
        "continue_on_success": _safe_getattr(env_unwrapped, "continue_on_success", None),
    }
    if any(value is not None for value in episode.values()):
        info["episode_reward"] = episode

    reset_constraints = {
        "map_xy_limit": _safe_getattr(env_unwrapped, "map_xy_limit", None),
        "wall_margin": _safe_getattr(env_unwrapped, "wall_margin", None),
        "goal_d_min": _safe_getattr(env_unwrapped, "goal_d_min", None),
        "goal_d_max": _safe_getattr(env_unwrapped, "goal_d_max", None),
        "safety_margin": _safe_getattr(env_unwrapped, "safety_margin", None),
        "max_reset_retries": _safe_getattr(env_unwrapped, "max_reset_retries", None),
        "obstacle_mode": _safe_getattr(env_unwrapped, "obstacle_mode", None),
    }
    if any(value is not None for value in reset_constraints.values()):
        info["reset_constraints"] = reset_constraints

    if hasattr(env_unwrapped, "obstacles"):
        try:
            obstacles = list(_safe_getattr(env_unwrapped, "obstacles", []))
            info["obstacles"] = {"count": len(obstacles), "items": obstacles}
        except Exception:
            info["obstacles"] = {"count": None, "items": None}

    ros_keys = [
        "scan_topic",
        "odom_topic",
        "cmd_vel_topic",
        "reset_world_service",
        "reset_sim_service",
        "set_model_state_service",
        "robot_model_name",
    ]
    ros_config = {key: _safe_getattr(env_unwrapped, key, None) for key in ros_keys if hasattr(env_unwrapped, key)}
    if ros_config:
        info["ros"] = ros_config

    return info


class RunRecorder:
    """Writes run_config.json (start) and run_summary.json (end) to the given data_dir."""
    def __init__(self, data_dir: str, algo_name: str, env_name: str, timestamp: str) -> None:
        self.data_dir = data_dir
        self.algo_name = algo_name
        self.env_name = env_name
        self.timestamp = timestamp
        os.makedirs(self.data_dir, exist_ok=True)

    def save_config(
        self,
        cfg: Any,
        agent: Any,
        env: Any,
        seed_global: int,
        script_params: Dict[str, Any],
        paths: Dict[str, str],
    ) -> str:
        cfg_dict = {}
        try:
            cfg_dict = vars(cfg)
        except Exception:
            # fallback
            cfg_dict = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}

        gymnasium_version = None
        gym_version = None
        try:
            import gymnasium
            gymnasium_version = getattr(gymnasium, "__version__", None)
        except Exception:
            pass
        try:
            import gym
            gym_version = getattr(gym, "__version__", None)
        except Exception:
            pass

        payload = {
            "timestamp": self.timestamp,
            "algo_name": self.algo_name,
            "env_name": self.env_name,
            "paths": paths,
            "seeds": {
                "cfg_seed": _safe_getattr(cfg, "seed", None),
                "seed_global": seed_global,
            },
            "script_params": script_params,
            "cfg": cfg_dict,
            "agent": _extract_agent_info(agent),
            "env": _extract_env_info(env),
            "runtime": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "gymnasium_version": gymnasium_version,
                "gym_version": gym_version,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }

        path = os.path.join(self.data_dir, "run_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
        return path

    def save_summary(
        self,
        total_steps: int,
        episodes_completed: int,
        duration_sec: float,
        final_model_path: str,
        metrics: Dict[str, Any],
    ) -> str:
        payload = {
            "timestamp": self.timestamp,
            "algo_name": self.algo_name,
            "env_name": self.env_name,
            "episodes_completed": episodes_completed,
            "total_steps": total_steps,
            "train_duration_sec": duration_sec,
            "final_model_path": final_model_path,
            "metrics": metrics,
            "config_path": os.path.join(self.data_dir, "run_config.json"),
        }

        path = os.path.join(self.data_dir, "run_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
        return path
