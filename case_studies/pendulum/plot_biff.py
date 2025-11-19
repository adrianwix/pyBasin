# --- Paste your PendulumODE code here ---

import matplotlib.pyplot as plt
import numpy as np
import torch
from pendulum_ode import PendulumODE, PendulumParams
from torchdiffeq import odeint  # type: ignore[import-untyped]

# Simulation parameters
alpha = 0.1  # damping coefficient
K = 1.0  # coupling strength
T_values = np.arange(0.01, 1.05, 0.05)  # Forcing values to sweep

dt = 0.01
num_steps = 10000
transient_index = 5000  # Number of steps to discard as transient
t = torch.linspace(0, dt * num_steps, num_steps)

bifurcation_data: list[tuple[float, float]] = []

# Sweep over the forcing parameter T
for T in T_values:
    print(f"Solving with T={T}")
    params: PendulumParams = {"alpha": alpha, "T": float(T), "K": K}
    ode_system = PendulumODE(params)
    y0 = torch.tensor([[0.5, 0.0]])  # Now shape is (1, 2)

    # Solve the ODE using torchdiffeq's odeint
    sol: torch.Tensor = odeint(ode_system.ode, y0, t)  # type: ignore
    sol_np = sol.detach().numpy()  # shape: (num_steps, 2)
    theta_series = sol_np[transient_index:, 0, 0]

    # Wrap theta into the interval [-pi, pi]
    theta_mod = ((theta_series + np.pi) % (2 * np.pi)) - np.pi

    # Record the theta values along with the current T value
    for theta_value in theta_mod:
        bifurcation_data.append((float(T), float(theta_value)))
    print(f"Completed solving for T={T}")

# Convert the bifurcation data to a NumPy array for plotting
bifurcation_array = np.array(bifurcation_data)

plt.figure(figsize=(10, 6))  # pyright: ignore[reportUnknownMemberType]
plt.scatter(bifurcation_array[:, 0], bifurcation_array[:, 1], s=0.1, color="black")  # pyright: ignore[reportUnknownMemberType]
plt.xlabel("T (forcing)")  # pyright: ignore[reportUnknownMemberType]
plt.ylabel("θ (mod 2π)")  # pyright: ignore[reportUnknownMemberType]
plt.title("Bifurcation Diagram for the Pendulum ODE using torchdiffeq")  # pyright: ignore[reportUnknownMemberType]
plt.show()  # pyright: ignore[reportUnknownMemberType]
