import numpy as np

class OrnsteinUhlenbeckNoise:
    """
    Ornstein–Uhlenbeck noise (paper Eq. 14/15/19) for exploration in continuous action spaces.
    """
    def __init__(self, action_dim, mu=None, b=0.15, sigma=None, dt=1.0, x0=None):
        self.action_dim = int(action_dim)
        self.mu = np.zeros(self.action_dim, dtype=np.float32) if mu is None else np.array(mu, dtype=np.float32)
        self.b = float(b)
        self.sigma = (
            np.full(self.action_dim, 0.2, dtype=np.float32)
            if sigma is None
            else (np.full(self.action_dim, float(sigma), dtype=np.float32) if isinstance(sigma, (int, float)) else np.array(sigma, dtype=np.float32))
        )
        self.dt = float(dt)
        self.x_prev = np.zeros(self.action_dim, dtype=np.float32) if x0 is None else np.array(x0, dtype=np.float32)

    def reset(self):
        self.x_prev = np.zeros(self.action_dim, dtype=np.float32)

    def sample(self):
        noise = np.random.normal(0.0, 1.0, size=self.action_dim).astype(np.float32)
        x = self.x_prev + self.b * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * noise
        self.x_prev = x
        return x
