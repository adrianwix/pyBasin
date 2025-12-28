# pyright: basic
"""
Standalone JAX/Diffrax benchmark for basin stability estimation of the pendulum system.

This benchmark implements the complete basin stability workflow using JAX and Diffrax
to measure pure JAX performance without the pybasin library overhead.

Key features:
- JAX/Diffrax ODE integration with GPU support
- Parallel integration of main samples and template solutions
- KNN classification using sklearn
- Complete basin stability computation
"""

import concurrent.futures
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve
from sklearn.neighbors import KNeighborsClassifier


class PendulumParams(NamedTuple):
    """Parameters for the pendulum ODE system."""

    alpha: float
    T: float
    K: float


def pendulum_dynamics(t, y, params: PendulumParams):
    """
    Pendulum ODE: dy/dt = f(t, y)
    y = [theta, theta_dot]

    Following the equations:
    dtheta/dt = theta_dot
    dtheta_dot/dt = -alpha * theta_dot + T - K * sin(theta)
    """
    theta, theta_dot = y
    dtheta_dt = theta_dot
    dtheta_dot_dt = -params.alpha * theta_dot + params.T - params.K * jnp.sin(theta)
    return jnp.array([dtheta_dt, dtheta_dot_dt])


def generate_grid_samples(n: int, min_limits: tuple, max_limits: tuple):
    """
    Generate grid samples for initial conditions.

    Args:
        n: Target number of samples (actual number may be slightly different)
        min_limits: (theta_min, theta_dot_min)
        max_limits: (theta_max, theta_dot_max)

    Returns:
        JAX array of shape (n_samples, 2) in float32 for faster GPU computation
    """
    n_per_dim = int(np.sqrt(n))

    theta_grid = jnp.linspace(min_limits[0], max_limits[0], n_per_dim, dtype=jnp.float32)
    theta_dot_grid = jnp.linspace(min_limits[1], max_limits[1], n_per_dim, dtype=jnp.float32)

    theta_mesh, theta_dot_mesh = jnp.meshgrid(theta_grid, theta_dot_grid, indexing="ij")
    samples = jnp.stack([theta_mesh.flatten(), theta_dot_mesh.flatten()], axis=1)

    return samples


def integrate_ode_batch(
    y0_batch: jax.Array, params: PendulumParams, t_span: tuple, n_steps: int, rtol=1e-6, atol=1e-8
):
    """
    ODE integration for a batch of initial conditions using Diffrax.

    Optimizations:
    - float32 precision for GPU efficiency
    - Relaxed tolerances (rtol=1e-6) to reduce steps

    Args:
        y0_batch: Initial conditions, shape (batch_size, 2)
        params: Pendulum parameters
        t_span: (t0, t1) time span
        n_steps: Number of evaluation points
        rtol: Relative tolerance (relaxed from 1e-8 to 1e-6)
        atol: Absolute tolerance

    Returns:
        t: Time points, shape (n_steps,)
        y: Solutions, shape (n_steps, batch_size, 2)
    """
    term = ODETerm(lambda t, y, args: pendulum_dynamics(t, y, params))
    solver = Tsit5()
    t0, t1 = t_span
    saveat = SaveAt(ts=jnp.linspace(t0, t1, n_steps, dtype=jnp.float32))
    stepsize_controller = PIDController(rtol=rtol, atol=atol)
    max_steps = 16**5

    def solve_single(y0):
        sol = diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=None,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )
        return sol.ys

    solve_batch = jax.vmap(solve_single)
    y_batch = solve_batch(y0_batch)

    if y_batch is None:
        raise RuntimeError("ODE integration failed, no solution returned.")
    y_batch = jnp.transpose(y_batch, (1, 0, 2))

    t = jnp.linspace(t0, t1, n_steps, dtype=jnp.float32)
    return t, y_batch


def extract_amplitudes(t: jax.Array, y: jax.Array):
    """
    Extract bifurcation amplitudes from trajectories.

    Args:
        t: Time array, shape (n_steps,)
        y: Trajectories, shape (n_steps, batch_size, 2)

    Returns:
        Amplitudes array, shape (batch_size, 2)
    """
    y_max = jnp.max(y, axis=0)
    y_min = jnp.min(y, axis=0)
    amplitudes = (y_max - y_min) / 2.0
    return amplitudes


def extract_features(t: jax.Array, y: jax.Array, time_steady: float):
    """
    Extract features from trajectories for classification.

    For the pendulum:
    - Fixed Point (FP): angular velocity is nearly constant (small oscillation)
    - Limit Cycle (LC): angular velocity varies significantly

    Args:
        t: Time array, shape (n_steps,)
        y: Trajectories, shape (n_steps, batch_size, 2)
        time_steady: Time after which system is in steady state

    Returns:
        Features array, shape (batch_size, 2) - one-hot encoded [FP, LC]
    """
    steady_mask = t >= time_steady
    y_steady = y[steady_mask]

    angular_velocity = y_steady[:, :, 1]

    delta = jnp.abs(jnp.max(angular_velocity, axis=0) - jnp.mean(angular_velocity, axis=0))

    is_fp = delta < 0.01

    features = jnp.where(
        is_fp[:, None],
        jnp.array([1.0, 0.0]),
        jnp.array([0.0, 1.0]),
    )

    return features


def compute_basin_stability(labels: np.ndarray):
    """
    Compute basin stability values from labels.

    Args:
        labels: Classification labels

    Returns:
        Dictionary of basin stability values
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    bs_vals = {
        str(label): count / total for label, count in zip(unique_labels, counts, strict=True)
    }
    return bs_vals


def main():
    """Main benchmark routine."""
    print("=" * 80)
    print("JAX/DIFFRAX PENDULUM BASIN STABILITY BENCHMARK")
    print("=" * 80)

    # Check device
    devices = jax.devices()
    device = devices[0]
    print(f"\nUsing device: {device}")
    print(f"Device kind: {device.device_kind}")

    total_start = time.perf_counter()

    print("\n" + "=" * 80)
    print("STEP 1: Setup Parameters")
    print("=" * 80)

    t1 = time.perf_counter()

    params = PendulumParams(alpha=0.1, T=0.5, K=1.0)
    print("\nPendulum parameters:")
    print(f"  alpha (damping): {params.alpha}")
    print(f"  T (torque): {params.T}")
    print(f"  K (stiffness): {params.K}")

    n_samples = 10000
    theta_min = -np.pi + np.arcsin(params.T / params.K)
    theta_max = np.pi + np.arcsin(params.T / params.K)
    theta_dot_min = -10.0
    theta_dot_max = 10.0

    print("\nSampling region:")
    print(f"  theta: [{theta_min:.4f}, {theta_max:.4f}]")
    print(f"  theta_dot: [{theta_dot_min:.4f}, {theta_dot_max:.4f}]")
    print(f"  Target samples: {n_samples}")

    t_span = (0.0, 1000.0)
    n_steps = 500
    time_steady = 950.0
    rtol = 1e-6
    atol = 1e-8

    print("\nIntegration parameters (optimized):")
    print(f"  Time span: {t_span}")
    print(f"  Evaluation points: {n_steps}")
    print(f"  Steady state time: {time_steady}")
    print(f"  rtol: {rtol}, atol: {atol}")
    print("  Precision: float32 (GPU optimized)")

    template_y0 = jnp.array([[0.5, 0.0], [2.7, 0.0]], dtype=jnp.float32)
    template_labels = ["FP", "LC"]

    print("\nTemplate initial conditions:")
    for label, y0 in zip(template_labels, template_y0, strict=True):
        print(f"  {label}: {y0}")

    t1_elapsed = time.perf_counter() - t1
    print(f"\nSetup complete in {t1_elapsed:.4f}s")

    print("\n" + "=" * 80)
    print("STEP 2: Generate Grid Samples")
    print("=" * 80)

    t2 = time.perf_counter()

    y0_grid = generate_grid_samples(
        n_samples, (theta_min, theta_dot_min), (theta_max, theta_dot_max)
    )
    actual_n = len(y0_grid)

    t2_elapsed = time.perf_counter() - t2
    print(f"\nGenerated {actual_n} grid samples in {t2_elapsed:.4f}s")
    print(f"Grid shape: {y0_grid.shape}")

    print("\n" + "=" * 80)
    print("STEP 3: Parallel ODE Integration (without JIT)")
    print("=" * 80)
    print("\nIntegrating both main grid and template solutions in parallel...")
    print("Note: No explicit JIT compilation - direct function calls")

    t3 = time.perf_counter()

    def integrate_fn(y0):
        return integrate_ode_batch(y0, params, t_span, n_steps, rtol, atol)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        main_future = executor.submit(lambda: integrate_fn(y0_grid))
        template_future = executor.submit(lambda: integrate_fn(template_y0))

        t_main, y_main = main_future.result()
        t_template, y_template = template_future.result()

        jax.block_until_ready(y_main)
        jax.block_until_ready(y_template)

    t3_elapsed = time.perf_counter() - t3
    print(f"\nBoth integrations complete in {t3_elapsed:.4f}s")
    print(f"  Main trajectory shape: {y_main.shape}")
    print(f"  Template trajectory shape: {y_template.shape}")

    print("\n" + "=" * 80)
    print("STEP 4: Extract Features")
    print("=" * 80)

    t4 = time.perf_counter()

    features_main = extract_features(t_main, y_main, time_steady)
    features_template = extract_features(t_template, y_template, time_steady)

    features_main_np = np.array(features_main)
    features_template_np = np.array(features_template)

    t4_elapsed = time.perf_counter() - t4
    print(f"\nFeature extraction complete in {t4_elapsed:.4f}s")
    print(f"  Main features shape: {features_main_np.shape}")
    print(f"  Template features shape: {features_template_np.shape}")

    print("\n" + "=" * 80)
    print("STEP 5: Classification")
    print("=" * 80)

    t5 = time.perf_counter()

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(features_template_np, template_labels)
    labels = knn.predict(features_main_np)

    t5_elapsed = time.perf_counter() - t5
    print(f"\nClassification complete in {t5_elapsed:.4f}s")
    print(f"  Predicted labels shape: {labels.shape}")

    print("\n" + "=" * 80)
    print("STEP 6: Compute Basin Stability")
    print("=" * 80)

    t6 = time.perf_counter()

    bs_vals = compute_basin_stability(labels)

    t6_elapsed = time.perf_counter() - t6
    print("\nBasin stability values:")
    for label, value in bs_vals.items():
        print(f"  {label}: {value:.6f}")

    print(f"\nComputation time: {t6_elapsed:.4f}s")

    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nTotal execution time: {total_elapsed:.4f}s")

    print("\nTiming Breakdown:")
    print(
        f"  1. Setup:              {t1_elapsed:8.4f}s  ({t1_elapsed / total_elapsed * 100:5.1f}%)"
    )
    print(
        f"  2. Sampling:           {t2_elapsed:8.4f}s  ({t2_elapsed / total_elapsed * 100:5.1f}%)"
    )
    print(
        f"  3. Integration:        {t3_elapsed:8.4f}s  ({t3_elapsed / total_elapsed * 100:5.1f}%)"
    )
    print(
        f"  4. Features:           {t4_elapsed:8.4f}s  ({t4_elapsed / total_elapsed * 100:5.1f}%)"
    )
    print(
        f"  5. Classification:     {t5_elapsed:8.4f}s  ({t5_elapsed / total_elapsed * 100:5.1f}%)"
    )
    print(
        f"  6. BS Computation:     {t6_elapsed:8.4f}s  ({t6_elapsed / total_elapsed * 100:5.1f}%)"
    )

    print("\n" + "=" * 80)
    print("Basin Stability Results:")
    for label, value in bs_vals.items():
        print(f"  {label}: {value:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
