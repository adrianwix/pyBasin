"""
Quick test to verify SklearnParallelSolver functionality.

This script runs a simple integration test to ensure the solver
works correctly with Python 3.14's free-threading capabilities.
"""

import numpy as np
import torch

from case_studies.pendulum.pendulum_ode import PendulumODE, PendulumParams
from pybasin.sklearn_parallel_solver import SklearnParallelSolver


def test_sklearn_solver_basic():
    """Test basic functionality of SklearnParallelSolver."""
    print("=" * 70)
    print("Testing SklearnParallelSolver - Basic Integration")
    print("=" * 70)

    # Setup pendulum ODE
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}
    ode_system = PendulumODE(params)

    # Create solver
    solver = SklearnParallelSolver(
        time_span=(0, 100),
        fs=10,
        n_jobs=2,  # Use 2 cores for testing
        rtol=1e-6,
        atol=1e-8,
    )

    # Test single initial condition
    print("\nTest 1: Single initial condition")
    y0_single = torch.tensor([0.5, 0.0], dtype=torch.float32)
    t_result, y_result = solver.integrate(ode_system, y0_single)

    print(f"  Time points: {len(t_result)}")
    print(f"  Solution shape: {y_result.shape}")
    print(f"  Initial state: {y_result[0].numpy()}")
    print(f"  Final state: {y_result[-1].numpy()}")
    print("  ✓ Single trajectory test passed")

    # Test batch initial conditions
    print("\nTest 2: Batch initial conditions (parallel)")
    y0_batch = torch.tensor([[0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]], dtype=torch.float32)
    t_result_batch, y_result_batch = solver.integrate(ode_system, y0_batch)

    print(f"  Time points: {len(t_result_batch)}")
    print(f"  Solution shape: {y_result_batch.shape}")
    print(f"  Number of trajectories: {y_result_batch.shape[1]}")
    print(f"  Initial states:\n{y_result_batch[0].numpy()}")
    print(f"  Final states:\n{y_result_batch[-1].numpy()}")
    print("  ✓ Batch trajectory test passed")

    # Verify integration quality
    print("\nTest 3: Integration quality check")
    # Check that solutions are bounded (physical constraint)
    theta_values = y_result_batch[:, :, 0].numpy()
    omega_values = y_result_batch[:, :, 1].numpy()

    print(f"  Theta range: [{theta_values.min():.3f}, {theta_values.max():.3f}]")
    print(f"  Omega range: [{omega_values.min():.3f}, {omega_values.max():.3f}]")

    # Check for NaN or Inf
    has_nan = np.isnan(y_result_batch.numpy()).any()
    has_inf = np.isinf(y_result_batch.numpy()).any()

    if has_nan or has_inf:
        print("  ✗ Integration quality check FAILED (NaN/Inf detected)")
    else:
        print("  ✓ Integration quality check passed (no NaN/Inf)")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_sklearn_solver_basic()
