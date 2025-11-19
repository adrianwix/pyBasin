# type: ignore
import numpy as np
from scipy.integrate import solve_ivp

N = 10_000  # number of IVPs
d = 2  # states per IVP (x, v)
t_span = (0.0, 10.0)

# Initial conditions shape: (N, d)
y0 = np.zeros((N, d))
y0[:, 0] = np.linspace(0.0, 1.0, N)  # different initial x, zero velocity


def rhs(t, y_flat):
    # y_flat: shape (N*d,)
    y = y_flat.reshape(N, d)
    x = y[:, 0]
    v = y[:, 1]

    # Example: x' = v, v' = -x  (N independent oscillators)
    dxdt = v
    dvdt = -x

    dydt = np.stack([dxdt, dvdt], axis=-1)  # shape (N, d)
    return dydt.ravel()  # shape (N*d,)


sol = solve_ivp(
    rhs,
    t_span,
    y0.ravel(),  # flattened ICs
    method="DOP853",  # or "RK45" for ode45-like
    rtol=1e-8,
    atol=1e-10,
    dense_output=False,
)

# sol.y has shape (N*d, n_times); reshape:
Y = sol.y.T.reshape(-1, N, d)  # (n_times, N, d)

print(Y)
