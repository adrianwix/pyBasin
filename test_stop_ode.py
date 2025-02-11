import torch
from torchdiffeq import odeint_event

# Define the Lorenz system


def lorenz(t, y):
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    y1, y2, y3 = y[..., 0], y[..., 1], y[..., 2]
    dy1 = sigma * (y2 - y1)
    dy2 = y1 * (rho - y3) - y2
    dy3 = y1 * y2 - beta * y3
    return torch.stack([dy1, dy2, dy3], dim=-1)

# Define the termination function


def lorenz_termination_event(t, y):
    """Stops integration if any state exceeds magnitude of 200."""
    return 200.0 - torch.max(torch.abs(y))


# Simulation parameters
t_start = torch.tensor([0.0])  # Initial time as a 1D tensor
t_end = 1000.0
fs = 25  # Sampling frequency in Hz
t_eval = torch.linspace(t_start.item(), t_end,
                        steps=int((t_end - t_start.item()) * fs))

# Multiple initial conditions
num_ics = 5  # Number of different initial conditions
y0 = torch.randn(num_ics, 3) * 5  # Random initial conditions

# Solve ODE with event handling
result = odeint_event(
    lorenz, y0, t_start, event_fn=lorenz_termination_event, t_eval=t_eval
)

# Unpack the result
t_values, y_values, event_t, event_y = result

# Output shape verification
print(f"t_values shape: {t_values.shape}")  # Time points
print(f"y_values shape: {y_values.shape}")  # Trajectories for all ICs
print(f"Event times: {event_t}")  # When the event occurred
print(f"Event states: {event_y}")  # The state at the event
