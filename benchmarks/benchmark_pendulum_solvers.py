"""
Benchmark and compare different ODE solvers for the pendulum system.

This script tests TorchDiffEqSolver vs TorchOdeSolver to find the fastest option.
Cache is always cleared before benchmarking to get accurate timing measurements.
"""

import shutil
import time
from pathlib import Path

import torch

from case_studies.pendulum.pendulum_ode import PendulumODE, PendulumParams
from pybasin.solver import TorchDiffEqSolver, TorchOdeSolver


def clear_cache():
    """Clear the solver cache directory."""
    # The cache is created relative to where this benchmark script is located
    cache_dir = Path(__file__).parent / "cache"

    if cache_dir.exists():
        print(f"Clearing cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("Cache cleared.\n")
    else:
        print(f"Cache directory does not exist: {cache_dir}")
        print("Will be created on first integration.\n")


def setup_pendulum_components(device):
    """Setup common pendulum system components."""
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}
    ode_system = PendulumODE(params)

    classifier_initial_conditions = torch.tensor(
        [
            [0.5, 0.0],  # FP: fixed point
            [2.7, 0.0],  # LC: limit cycle
        ],
        dtype=torch.float32,
        device=device,
    )

    return ode_system, classifier_initial_conditions


def benchmark_solver(solver_name: str, solver, ode_system, initial_conditions):
    """Benchmark a single solver."""
    print(f"{'=' * 80}")
    print(f"TESTING: {solver_name}")
    print(f"{'=' * 80}")
    print(f"Device: {solver.device}")
    print(f"Number of templates: {len(initial_conditions)}")
    print(f"Integration time span: {solver.time_span}")
    print(f"Number of time steps: {solver.n_steps}")

    if hasattr(solver, "method"):
        print(f"Method: {solver.method}")
    if hasattr(solver, "use_jit"):
        print(f"JIT enabled: {solver.use_jit}")

    print("\nIntegrating template initial conditions...")
    start_time = time.perf_counter()

    with torch.no_grad():
        t, y = solver.integrate(ode_system, initial_conditions)

    integration_time = time.perf_counter() - start_time

    print(f"\n{'=' * 80}")
    print(f"RESULTS - {solver_name}")
    print(f"{'=' * 80}")
    print(f"Integration time: {integration_time:.3f} seconds")
    print(f"Solution shape:   {y.shape}")
    print()

    return integration_time


def main():
    """Main benchmark function comparing different solvers."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning benchmarks on device: {device}\n")

    # Setup common components
    ode_system, initial_conditions = setup_pendulum_components(device)

    results = {}

    # Test configurations
    configs = [
        {
            "name": "TorchDiffEqSolver (n_steps=100)",
            "solver": TorchDiffEqSolver(time_span=(0, 1000), n_steps=100, device=device),
        },
        {
            "name": "TorchDiffEqSolver (n_steps=500)",
            "solver": TorchDiffEqSolver(time_span=(0, 1000), n_steps=500, device=device),
        },
        {
            "name": "TorchOdeSolver dopri5 (n_steps=100, no JIT)",
            "solver": TorchOdeSolver(
                time_span=(0, 1000), n_steps=100, device=device, method="dopri5", use_jit=False
            ),
        },
        {
            "name": "TorchOdeSolver dopri5 (n_steps=100, JIT)",
            "solver": TorchOdeSolver(
                time_span=(0, 1000), n_steps=100, device=device, method="dopri5", use_jit=True
            ),
        },
        {
            "name": "TorchOdeSolver dopri5 (n_steps=500, no JIT)",
            "solver": TorchOdeSolver(
                time_span=(0, 1000), n_steps=500, device=device, method="dopri5", use_jit=False
            ),
        },
        {
            "name": "TorchOdeSolver dopri5 (n_steps=500, JIT)",
            "solver": TorchOdeSolver(
                time_span=(0, 1000), n_steps=500, device=device, method="dopri5", use_jit=True
            ),
        },
        {
            "name": "TorchOdeSolver tsit5 (n_steps=100, no JIT)",
            "solver": TorchOdeSolver(
                time_span=(0, 1000), n_steps=100, device=device, method="tsit5", use_jit=False
            ),
        },
        {
            "name": "TorchOdeSolver tsit5 (n_steps=100, JIT)",
            "solver": TorchOdeSolver(
                time_span=(0, 1000), n_steps=100, device=device, method="tsit5", use_jit=True
            ),
        },
    ]

    for config in configs:
        clear_cache()
        elapsed = benchmark_solver(config["name"], config["solver"], ode_system, initial_conditions)
        results[config["name"]] = elapsed

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - All Solvers")
    print("=" * 80)
    print(f"{'Solver':<45} {'Time (s)':>10} {'Speedup':>10}")
    print("-" * 80)

    baseline = results[list(results.keys())[0]]
    for name, elapsed in results.items():
        speedup = baseline / elapsed
        print(f"{name:<45} {elapsed:>10.3f} {speedup:>10.2f}x")

    # Find fastest
    fastest = min(results, key=results.get)  # type: ignore
    print("\n" + "=" * 80)
    print(f"FASTEST: {fastest}")
    print(f"Time: {results[fastest]:.3f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
