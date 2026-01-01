

import collections
import random
import numpy as np
 
class ReplayBuffer:
    def __init__(self, capacity, seed: int | None = None):
        self.buffer = collections.deque(maxlen=capacity)
        # 私有 RNG：避免被全局 random 其它调用污染
        self.rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        """按 episode 重置采样 RNG，确保严格可复现。"""
        self.rng.seed(int(seed))
 
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
 
    def sample(self, batch_size):
        transitions = self.rng.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
 
    def size(self):
        return len(self.buffer)
