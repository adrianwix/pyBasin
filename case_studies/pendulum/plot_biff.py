import torch
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# --- Paste your PendulumODE code here ---
from typing import TypedDict
from case_study_pendulum.PendulumODE import PendulumODE
from pybasin.ode_system import ODESystem  # Ensure pybasin is installed


# Simulation parameters
alpha = 0.1  # damping coefficient
K = 1.0  # coupling strength
T_values = np.arange(0.01, 1.05, 0.05)  # Forcing values to sweep

dt = 0.01
num_steps = 10000
transient_index = 5000  # Number of steps to discard as transient
t = torch.linspace(0, dt * num_steps, num_steps)

bifurcation_data = []

# Sweep over the forcing parameter T
for T in T_values:
    print(f"Solving with T={T}")
    params = {"alpha": alpha, "T": T, "K": K}
    ode_system = PendulumODE(params)
    y0 = torch.tensor([[0.5, 0.0]])  # Now shape is (1, 2)

    # Solve the ODE using torchdiffeq's odeint
    sol = odeint(ode_system.ode, y0, t)
    sol_np = sol.detach().numpy()  # shape: (num_steps, 2)
    theta_series = sol_np[transient_index:, 0, 0]

    # Wrap theta into the interval [-pi, pi]
    theta_mod = ((theta_series + np.pi) % (2 * np.pi)) - np.pi

    # Record the theta values along with the current T value
    for theta_value in theta_mod:
        bifurcation_data.append((T, float(theta_value)))
    print(f"Completed solving for T={T}")

# Convert the bifurcation data to a NumPy array for plotting
bifurcation_data = np.array(bifurcation_data)

plt.figure(figsize=(10, 6))
plt.scatter(bifurcation_data[:, 0], bifurcation_data[:, 1], s=0.1, color="black")
plt.xlabel("T (forcing)")
plt.ylabel("θ (mod 2π)")
plt.title("Bifurcation Diagram for the Pendulum ODE using torchdiffeq")
plt.show()
