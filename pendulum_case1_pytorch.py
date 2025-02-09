import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import time  # For measuring execution time

# import os

# # This makes CUDA think there are no GPUs available
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
# -----------------------------------------------------
# 1. Define the RandomSampler class
# -----------------------------------------------------
class RandomSampler:
    """Generates random samples using a uniform distribution within the specified range."""
    def __init__(self, min_limits, max_limits, state_dim=2):
        self.min_limits = np.array(min_limits)
        self.max_limits = np.array(max_limits)
        self.state_dim = state_dim
        
    def sample(self, N: int) -> np.ndarray:
        rng = np.random.default_rng()
        return rng.uniform(self.min_limits, self.max_limits, (N, self.state_dim))

# -----------------------------------------------------
# 2. Define the pendulum ODE as a PyTorch module.
#    (State y: [theta, theta_dot])
# -----------------------------------------------------
class PendulumODE(nn.Module):
    def __init__(self, alpha, T, K):
        super(PendulumODE, self).__init__()
        self.alpha = alpha
        self.T = T
        self.K = K

    def forward(self, t, y):
        # y is of shape [batch_size, 2]
        theta = y[:, 0]
        theta_dot = y[:, 1]
        dtheta_dt = theta_dot
        dtheta_dot_dt = -self.alpha * theta_dot + self.T - self.K * torch.sin(theta)
        return torch.stack([dtheta_dt, dtheta_dot_dt], dim=1)

# -----------------------------------------------------
# 3. Set Parameters and Create the Sampler
# -----------------------------------------------------
# Pendulum parameters
alpha = 0.1
T = 0.5
K = 1.0

# Instantiate the ODE system
ode_system = PendulumODE(alpha, T, K)

# For fixed points, we need T - K*sin(theta) = 0  =>  sin(theta) = T/K.
# With T=0.5 and K=1.0, sin(theta)=0.5 so theta ~ 0.5236 or ~2.618 (in radians).
# We define our sampling range for theta accordingly.
min_theta = -np.pi + np.arcsin(T / K)
max_theta = np.pi + np.arcsin(T / K)
# For theta_dot, we use:
min_theta_dot = -10.0
max_theta_dot = 10.0

# Create the sampler
sampler = RandomSampler(min_limits=[min_theta, min_theta_dot],
                        max_limits=[max_theta, max_theta_dot],
                        state_dim=2)

# Number of initial conditions to sample
N = 1000
initial_conditions_np = sampler.sample(N)
# Convert to a torch tensor (using float64 for precision)
initial_conditions = torch.tensor(initial_conditions_np, dtype=torch.float64)

# -----------------------------------------------------
# 4. Define the Time Span and Solve the ODE in Batch
# -----------------------------------------------------
t0, t_final = 0.0, 1000.0
num_time_points = 1000  # Increase if higher time resolution is needed.
t = torch.linspace(t0, t_final, num_time_points, dtype=torch.float64)

# -----------------------------------------------------
# 5. Measure the Execution Time
# -----------------------------------------------------
start_time = time.time()  # Start timer

# Solve the ODE using torchdiffeq's odeint.
# The solution tensor has shape [num_time_points, N, 2]
solution = odeint(ode_system, initial_conditions, t, method='dopri5', rtol=1e-6, atol=1e-6)

# -----------------------------------------------------
# 6. Analyze the Solution for Basin Stability
# -----------------------------------------------------
# Convert the solution to a NumPy array for analysis.
solution_np = solution.cpu().detach().numpy()

# Extract the angular velocity (theta_dot) trajectories.
# solution_np has shape [num_time_points, N, 2]; index 1 corresponds to theta_dot.
theta_dot_trajs = solution_np[:, :, 1]  # Shape: [num_time_points, N]

# Analyze only the tail of the simulation (e.g., last 10% of time points)
tail_start = int(num_time_points * 0.9)
tail = theta_dot_trajs[tail_start:, :]  # Shape: [tail_points, N]

# For each simulation, compute the variation in theta_dot in the tail:
# Here, we use the difference between the maximum and minimum values.
delta = np.abs(np.max(tail, axis=0) - np.min(tail, axis=0))

# Set a threshold below which the variation is considered negligible (indicating a fixed point).
threshold = 1e-2

# Classify each simulation:
# If delta < threshold, we say it has reached a fixed point.
is_fixed_point = delta < threshold

# Compute basin stability:
basin_stability_fixed = np.mean(is_fixed_point)
basin_stability_limit_cycle = 1 - basin_stability_fixed

# Stop the timer after analysis
end_time = time.time()
total_time = end_time - start_time

# -----------------------------------------------------
# 7. Output the Results and Execution Time
# -----------------------------------------------------
print("Basin stability for fixed point: {:.3f}".format(basin_stability_fixed))
print("Basin stability for limit cycle: {:.3f}".format(basin_stability_limit_cycle))
print("Total execution time: {:.3f} seconds".format(total_time))
