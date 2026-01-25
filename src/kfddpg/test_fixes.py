import torch
import numpy as np
import sys
import os

# Adapt path to import modules
sys.path.append("/home/lst/kfdqn_catkin_ws/src/kfddpg")

from agents.utils import OrnsteinUhlenbeckNoise
from agents.kfddpg import KFDDPGAgent
from agents.fuzzy_system import FuzzySystem

def test_ou_noise():
    print("Testing OU Noise...")
    noise = OrnsteinUhlenbeckNoise(action_dim=2, mu=0, b=0.15, sigma=0.0, dt=1.0, x0=[1.0, 1.0])
    # With sigma=0, x_{t+1} = x_t + b(mu - x_t) = x_t + 0.15 * (0 - x_t) = 0.85 * x_t
    # x0 = 1.0 -> x1 = 0.85
    sample = noise.sample()
    print(f"Sample with sigma=0, x0=1.0: {sample}")
    if np.allclose(sample, [0.85, 0.85]):
        print("PASS: OU Noise deterministic decay check.")
    else:
        print("FAIL: OU Noise check.")

def test_theta_g_learnable():
    print("\nTesting Theta_G Learnable...")
    agent = KFDDPGAgent(state_dim=94, action_dim=2, env_name="Env2")
    
    # Check if theta_g_logit is a parameter
    if isinstance(agent.theta_g_logit, torch.nn.Parameter):
        print("PASS: theta_g_logit is nn.Parameter.")
    else:
        print("FAIL: theta_g_logit is NOT nn.Parameter.")
        
    # Check optimizer
    if agent.theta_g_optimizer is not None:
        print("PASS: theta_g_optimizer exists.")
    else:
        print("FAIL: No optimizer for theta_g.")

def test_fuzzy_rules():
    print("\nTesting Fuzzy System Side Rules...")
    device = torch.device('cpu')
    fs = FuzzySystem(device, "Env2") # Env2 is ObstacleAvoid
    
    # Construct a state:
    # 0-90: Lidar. Let's make Left close, others far.
    # Left Sector is indices [23:45]. Let's set index 30 to 0.1 (Very Close).
    # All others 1.0 (Far).
    # Theta: 0 (Front).
    
    state = torch.ones(1, 94)
    state[0, 30] = 0.05 # Left Close
    state[0, 90] = 0.0 # Theta Front
    
    # Forward
    action_scores = fs(state) 
    # Action dim 3: [Right, Left, Fwd]
    # Expect: Turn Right (index 0) to be high.
    
    print(f"State: Left Obstacle (0.05). Scores: {action_scores.detach().numpy()}")
    
    # Check if Index 0 (Turn Right) is highest or significantly activated
    if action_scores[0, 0] > action_scores[0, 1]:
        print("PASS: Left Obstacle -> Turn Right score > Turn Left score.")
    else:
        print("FAIL: Left Obstacle did not trigger Turn Right dominance.")
        
    # Test Right Obstacle
    state_r = torch.ones(1, 94)
    state_r[0, 60] = 0.05 # Right Close (Index ~60 is in [45:67])
    state_r[0, 90] = 0.0
    
    action_scores_r = fs(state_r)
    print(f"State: Right Obstacle (0.05). Scores: {action_scores_r.detach().numpy()}")
    
    if action_scores_r[0, 1] > action_scores_r[0, 0]:
        print("PASS: Right Obstacle -> Turn Left score > Turn Right score.")
    else:
        print("FAIL: Right Obstacle did not trigger Turn Left dominance.")

if __name__ == "__main__":
    test_ou_noise()
    test_theta_g_learnable()
    test_fuzzy_rules()
