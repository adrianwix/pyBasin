from typing import List, Dict
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

params = {"alpha": 0.1, "T": 0.5, "K": 1.0}
y0 = [0.0, 0.0]  # Initial conditions: [theta, theta_dot]
y0 = [-2.0, 6.0]  # Initial conditions: [theta, theta_dot]

t_span = (0, 1000)  # Time span for the integration

OHE = {
    "FP": np.array([1.0, 0.0], dtype=np.float64),
    "LC": np.array([0.0, 1.0], dtype=np.float64)
}


def pendulum_ode(t: float, y: NDArray[np.float64], params: Dict) -> List[float]:
    """
    Right-hand side (RHS) for the pendulum ODE.

    Parameters
    ----------
    t : float
        The current time (not explicitly used if the system is time-invariant).
    y : NDArray[np.float64], shape (2,)
        The current state, [theta, theta_dot].
    params : Params
        Dictionary of model parameters, for example {"alpha": 0.1, "T": 0.5, "K": 1.0}.

    Returns
    -------
    dydt : List[float]
        The derivatives: [dtheta/dt, dtheta_dot/dt].
    """
    alpha = params["alpha"]
    T = params["T"]
    K = params["K"]

    theta, theta_dot = y
    dtheta_dt = theta_dot
    dtheta_dot_dt = -alpha * theta_dot + T - K * np.sin(theta)

    return [dtheta_dt, dtheta_dot_dt]


def features_pendulum(
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    steady_state_time: float = 950.0
) -> NDArray[np.float64]:
    """
    Replicates the MATLAB 'features_pendulum' function in Python:
      1) Identify time indices for t > steady_state_time (steady-state).
      2) Compute Delta = |max(theta_dot) - mean(theta_dot)|.
      3) If Delta < 0.01 => [1,0] (FP), else => [0,1] (LC).

    Parameters
    ----------
    t : NDArray[np.float64]
        Time values from the integration.
    y : NDArray[np.float64], shape (len(t), 2)
        The states at each time in t.  y[:,0] = theta, y[:,1] = theta_dot
    steady_state_time : float
        Time after which we consider the system to be near steady-state.

    Returns
    -------
    X : NDArray[np.float64], shape (2, 1)
        A one-hot vector, [1,0]^T for FP or [0,1]^T for LC.
    """
    # Indices where t > steady_state_time
    idx_steady = np.where(t > steady_state_time)[0]
    if len(idx_steady) == 0:
        print("Warning: No steady state found.")
        # If we never get beyond steady_state_time, default to [1,0]
        return np.array(OHE["FP"], dtype=np.float64)

    print(f"Steady state found at t={t[idx_steady[0]]}")
    idx_start = idx_steady[0]

    # Use the second state (theta_dot) for the portion after steady_state_time
    portion = y[idx_start:, 1]
    delta = np.abs(np.max(portion) - np.mean(portion))
    print(f"Delta = {delta}")

    if delta < 0.01:
        print("Fixed Point (FP)")
        # FP (Fixed Point)
        return np.array(OHE["FP"], dtype=np.float64)
    else:
        # LC (Limit Cycle)
        return np.array(OHE["LC"], dtype=np.float64)


sol = solve_ivp(
    fun=lambda t, y: pendulum_ode(t, y, params),
    t_span=t_span,
    y0=y0,
    method="RK45",
    rtol=1e-8,
    dense_output=True
)


# Plot the results
t = sol.t
theta = sol.y[0]

y = sol.y.T  # shape => (len(t), 2)
X_i = features_pendulum(t, y, steady_state_time=950)

theta_dot = sol.y[1]

plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(t, theta, label='theta (angle)')
plt.xlabel('Time')
plt.ylabel('Theta')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, theta_dot, label='theta_dot (angular velocity)', color='r')
plt.xlabel('Time')
plt.ylabel('Theta_dot')
plt.legend()

plt.tight_layout()
plt.show()
