# src/scripts/utils/exploration.py

def get_linear_decay_epsilon(current_idx, cfg):
    """
    计算线性衰减的 Epsilon 值
    :param current_idx: 当前计数（这里指当前的总步数 total_steps）
    :param cfg: 配置对象，包含 decay_start, decay_steps (现在代表步数), epsilon_start, epsilon_end
    """
    # 1. 预热期: 在 decay_start 步之前，保持 epsilon_start
    if current_idx < cfg.decay_start:
        return cfg.epsilon_start
    
    # 2. 计算衰减过程
    steps_done = current_idx - cfg.decay_start
    
    # 防止除以0
    if cfg.decay_steps <= 0:
        return cfg.epsilon_end
        
    # 3. 计算进度 (0.0 -> 1.0)
    # 此时 cfg.decay_steps 代表衰减持续的总步数
    progress = min(1.0, steps_done / cfg.decay_steps)
    
    # 4. 计算当前 Epsilon
    epsilon_range = cfg.epsilon_start - cfg.epsilon_end
    current_epsilon = cfg.epsilon_start - (epsilon_range * progress)
    
    return max(cfg.epsilon_end, current_epsilon)