# pyright: basic
"""
Experiment: Parameter batching for ODE integration.

Tests batching over parameter grids with a single initial condition using:
1. Diffrax (JAX) with vmap
2. torchdiffeq (PyTorch) with vectorized ODE

Uses the pendulum ODE as the test case:
    dθ/dt = θ̇
    dθ̇/dt = -α·θ̇ + T - K·sin(θ)
"""

import time

import jax
import jax.numpy as jnp
import torch
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve
from torchdiffeq import odeint

ArrayLike = jax.typing.ArrayLike


def run_diffrax_parameter_batching() -> None:
    """Test parameter batching with Diffrax using vmap."""
    print("=" * 60)
    print("Diffrax Parameter Batching (JAX + vmap)")
    print("=" * 60)

    def pendulum_ode(
        t: ArrayLike, y: jax.Array, args: tuple[jax.Array, jax.Array, jax.Array]
    ) -> jax.Array:
        alpha, torque, k = args
        theta, theta_dot = y[0], y[1]
        dtheta = theta_dot
        dtheta_dot = -alpha * theta_dot + torque - k * jnp.sin(theta)
        return jnp.array([dtheta, dtheta_dot])

    def solve_single(params: jax.Array) -> jax.Array:
        alpha, torque, k = params[0], params[1], params[2]
        term = ODETerm(lambda t, y, _: pendulum_ode(t, y, (alpha, torque, k)))
        saveat = SaveAt(ts=jnp.linspace(0, 10, 100))
        sol = diffeqsolve(
            term,
            Tsit5(),
            t0=0.0,
            t1=10.0,
            dt0=0.01,
            y0=jnp.array([0.1, 0.0]),
            saveat=saveat,
        )
        assert isinstance(sol.ys, jax.Array)
        return sol.ys

    alphas = jnp.linspace(0.1, 1.0, 5)
    torques = jnp.linspace(0.0, 2.0, 5)
    k_fixed = 1.0

    alpha_grid, torque_grid = jnp.meshgrid(alphas, torques, indexing="ij")
    param_grid = jnp.stack(
        [alpha_grid.flatten(), torque_grid.flatten(), jnp.full(25, k_fixed)],
        axis=1,
    )

    print(f"Parameter grid shape: {param_grid.shape}")
    print(f"  - alphas: {alphas}")
    print(f"  - torques: {torques}")
    print(f"  - K (fixed): {k_fixed}")

    batched_solve = jax.vmap(solve_single)

    print("\nCompiling (first run)...")
    start = time.perf_counter()
    trajectories = batched_solve(param_grid)
    trajectories.block_until_ready()
    compile_time = time.perf_counter() - start
    print(f"Compile + first run time: {compile_time:.3f}s")

    print("\nRunning batched solve (compiled)...")
    start = time.perf_counter()
    trajectories = batched_solve(param_grid)
    trajectories.block_until_ready()
    run_time = time.perf_counter() - start

    print(f"Run time: {run_time:.3f}s")
    print(f"Output shape: {trajectories.shape}")
    print("  - (n_param_sets, n_timesteps, state_dim)")

    print("\nSample final states (first 5 parameter sets):")
    for i in range(5):
        final_state = trajectories[i, -1, :]
        print(
            f"  params[{i}] (α={param_grid[i, 0]:.2f}, T={param_grid[i, 1]:.2f}): "
            f"θ={final_state[0]:.4f}, θ̇={final_state[1]:.4f}"
        )


class BatchedPendulumTorch(torch.nn.Module):
    """Vectorized pendulum ODE for torchdiffeq parameter batching."""

    def __init__(self, alphas: torch.Tensor, torques: torch.Tensor, ks: torch.Tensor):
        super().__init__()
        self.alphas = alphas
        self.torques = torques
        self.ks = ks

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        theta = y[:, 0]
        theta_dot = y[:, 1]

        dtheta = theta_dot
        dtheta_dot = -self.alphas * theta_dot + self.torques - self.ks * torch.sin(theta)

        return torch.stack([dtheta, dtheta_dot], dim=1)


def run_torchdiffeq_parameter_batching() -> None:
    """Test parameter batching with torchdiffeq using vectorized ODE."""
    print("\n" + "=" * 60)
    print("torchdiffeq Parameter Batching (PyTorch)")
    print("=" * 60)

    alphas = torch.linspace(0.1, 1.0, 5)
    torques = torch.linspace(0.0, 2.0, 5)
    k_fixed = 1.0

    alpha_grid, torque_grid = torch.meshgrid(alphas, torques, indexing="ij")
    alpha_flat = alpha_grid.flatten()
    torque_flat = torque_grid.flatten()
    k_flat = torch.full((25,), k_fixed)

    n_params = alpha_flat.shape[0]

    print(f"Parameter grid shape: ({n_params}, 3)")
    print(f"  - alphas: {alphas.tolist()}")
    print(f"  - torques: {torques.tolist()}")
    print(f"  - K (fixed): {k_fixed}")

    y0_single = torch.tensor([0.1, 0.0])
    y0 = y0_single.unsqueeze(0).expand(n_params, -1).clone()

    print(f"Initial condition (broadcast): {y0_single.tolist()} -> shape {y0.shape}")

    ode = BatchedPendulumTorch(alpha_flat, torque_flat, k_flat)
    t_span = torch.linspace(0, 10, 100)

    print("\nRunning batched solve...")
    start = time.perf_counter()
    trajectories_result = odeint(ode, y0, t_span, method="dopri5")
    assert isinstance(trajectories_result, torch.Tensor)
    trajectories = trajectories_result
    run_time = time.perf_counter() - start

    print(f"Run time: {run_time:.3f}s")
    print(f"Output shape: {trajectories.shape}")
    print("  - (n_timesteps, n_param_sets, state_dim)")

    trajectories_reordered = trajectories.permute(1, 0, 2)
    print(f"Reordered shape: {trajectories_reordered.shape}")
    print("  - (n_param_sets, n_timesteps, state_dim)")

    print("\nSample final states (first 5 parameter sets):")
    for i in range(5):
        final_state = trajectories_reordered[i, -1, :]
        print(
            f"  params[{i}] (α={alpha_flat[i]:.2f}, T={torque_flat[i]:.2f}): "
            f"θ={final_state[0]:.4f}, θ̇={final_state[1]:.4f}"
        )


def compare_results() -> None:
    """Compare results from both backends."""
    print("\n" + "=" * 60)
    print("Comparing Results (JAX vs PyTorch)")
    print("=" * 60)

    alphas_jax = jnp.linspace(0.1, 1.0, 3)
    torques_jax = jnp.linspace(0.0, 1.0, 3)
    k_fixed = 1.0

    alpha_grid_jax, torque_grid_jax = jnp.meshgrid(alphas_jax, torques_jax, indexing="ij")
    param_grid_jax = jnp.stack(
        [alpha_grid_jax.flatten(), torque_grid_jax.flatten(), jnp.full(9, k_fixed)],
        axis=1,
    )

    def pendulum_ode_jax(
        t: ArrayLike, y: jax.Array, args: tuple[jax.Array, jax.Array, jax.Array]
    ) -> jax.Array:
        alpha, torque, k = args
        theta, theta_dot = y[0], y[1]
        dtheta = theta_dot
        dtheta_dot = -alpha * theta_dot + torque - k * jnp.sin(theta)
        return jnp.array([dtheta, dtheta_dot])

    def solve_single_jax(params: jax.Array) -> jax.Array:
        alpha, torque, k = params[0], params[1], params[2]
        term = ODETerm(lambda t, y, _: pendulum_ode_jax(t, y, (alpha, torque, k)))
        saveat = SaveAt(ts=jnp.linspace(0, 10, 50))
        sol = diffeqsolve(
            term,
            Tsit5(),
            t0=0.0,
            t1=10.0,
            dt0=0.01,
            y0=jnp.array([0.1, 0.0]),
            saveat=saveat,
        )
        assert isinstance(sol.ys, jax.Array)
        return sol.ys

    batched_solve_jax = jax.vmap(solve_single_jax)
    traj_jax = batched_solve_jax(param_grid_jax)

    alphas_torch = torch.linspace(0.1, 1.0, 3)
    torques_torch = torch.linspace(0.0, 1.0, 3)

    alpha_grid_torch, torque_grid_torch = torch.meshgrid(alphas_torch, torques_torch, indexing="ij")
    alpha_flat = alpha_grid_torch.flatten()
    torque_flat = torque_grid_torch.flatten()
    k_flat = torch.full((9,), k_fixed)

    y0 = torch.tensor([[0.1, 0.0]]).expand(9, -1).clone()
    ode_torch = BatchedPendulumTorch(alpha_flat.double(), torque_flat.double(), k_flat.double())
    t_span = torch.linspace(0, 10, 50, dtype=torch.float64)
    traj_torch_result = odeint(ode_torch, y0.double(), t_span, method="dopri5")
    assert isinstance(traj_torch_result, torch.Tensor)
    traj_torch = traj_torch_result.permute(1, 0, 2)

    traj_jax_np = jax.device_get(traj_jax)
    traj_torch_np = traj_torch.numpy()

    max_diff = abs(traj_jax_np - traj_torch_np).max()
    mean_diff = abs(traj_jax_np - traj_torch_np).mean()

    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")

    if max_diff < 1e-4:
        print("✓ Results match within tolerance!")
    else:
        print("⚠ Results differ significantly - check solver settings")


if __name__ == "__main__":
    run_diffrax_parameter_batching()
    run_torchdiffeq_parameter_batching()
    compare_results()
