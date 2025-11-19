# type: ignore
"""
Benchmark Scipy ODE solvers (DOP853, RK45) for damped driven pendulum
Measures ODE integration performance and computes basin stability values.

Based on SklearnParallelSolver architecture from solver.py but without parallel processing
to measure single-threaded scipy performance.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from basin_stability_utils import (
    classify_pendulum_trajectories,
    compute_basin_stability,
    print_basin_stability_results,
    print_verification_results,
    verify_against_reference,
)


def ode_pendulum(t, y, alpha, capital_t, capital_k):
    """
    Damped driven pendulum ODE system (Scipy version)

    dy/dt = [y[1], -alpha*y[1] + capital_t - capital_k*sin(y[0])]

    Parameters:
        t: time (scalar)
        y: state vector [phi, omega]
        alpha: dissipation coefficient
        capital_t: constant angular acceleration
        capital_k: stiffness coefficient (g/l)

    Returns:
        dydt: derivative vector
    """
    phi, omega = y
    dphi_dt = omega
    domega_dt = -alpha * omega + capital_t - capital_k * np.sin(phi)
    return np.array([dphi_dt, domega_dt])


def get_git_commit():
    """Get current git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_benchmark(config, method="DOP853", max_time=120):
    """
    Run integration benchmark for all initial conditions using scipy.integrate.solve_ivp.

    This follows the SklearnParallelSolver architecture from solver.py but runs
    serially to measure single-threaded scipy performance.

    Parameters:
        config: Configuration dictionary
        method: Scipy solver method ('DOP853', 'RK45', 'RK23', 'Radau', 'BDF', 'LSODA')
        max_time: Maximum time in seconds before timeout
    """
    print("=" * 50)
    print(f"Scipy {method} Integration Benchmark")
    print("=" * 50)
    print(f"System: {config['system']['name']}")
    print(f"Number of samples: {config['initial_conditions']['n_samples']}")
    print(
        f"Time span: [{config['time_integration']['t_start']}, {config['time_integration']['t_end']}]"
    )
    print(f"Solver: {method}")
    print(f"Device: CPU (scipy is CPU-only)")
    print("=" * 50)
    print()

    # System parameters
    alpha = config["ode_parameters"]["alpha"]
    capital_t = config["ode_parameters"]["T"]
    capital_k = config["ode_parameters"]["K"]

    # Time integration settings
    t_start = config["time_integration"]["t_start"]
    t_end = config["time_integration"]["t_end"]
    t_span = (t_start, t_end)
    rtol = config["time_integration"]["rtol"]
    atol = config["time_integration"]["atol"]

    # Calculate max_step similar to SklearnParallelSolver
    max_step = (t_end - t_start) / 100

    # Generate time evaluation points
    # Use n_steps from config if available, otherwise default to 500
    n_steps = config.get("time_integration", {}).get("n_steps", 500)
    t_eval = np.linspace(t_start, t_end, n_steps)

    # Generate initial conditions
    n_samples = config["initial_conditions"]["n_samples"]
    roi_min = np.array(config["initial_conditions"]["roi_min"])
    roi_max = np.array(config["initial_conditions"]["roi_max"])
    seed = config["initial_conditions"]["random_seed"]

    np.random.seed(seed)
    dof = config["system"]["dof"]
    ic_grid = np.random.uniform(roi_min, roi_max, (n_samples, dof))

    print(f"Generated {n_samples} initial conditions (seed={seed})")
    print(f"ROI: [{roi_min[0]:.4f}, {roi_max[0]:.4f}] x [{roi_min[1]:.4f}, {roi_max[1]:.4f}]")
    print(f"Sample IC[0]: [{ic_grid[0, 0]:.4f}, {ic_grid[0, 1]:.4f}]")
    print(f"Sample IC[-1]: [{ic_grid[-1, 0]:.4f}, {ic_grid[-1, 1]:.4f}]")
    print(f"Time evaluation points: {n_steps}")
    print(f"Max step size: {max_step:.4f}")
    print()

    # Define ODE function wrapper (similar to SklearnParallelSolver)
    def ode_func(t, y):
        """Wrapper for ODE function to match scipy.integrate.solve_ivp signature"""
        return ode_pendulum(float(t), np.asarray(y), alpha, capital_t, capital_k)

    # Run benchmark - serial execution similar to SklearnParallelSolver single trajectory mode
    print(f"Starting integration (method={method}, max_time={max_time}s)...")

    # Store results: (n_steps, n_samples, dof)
    results_list = []
    integration_complete = True
    completed_count = 0

    start_time = time.perf_counter()

    try:
        for i in range(n_samples):
            if (i + 1) % 1000 == 0:
                print(f"Progress: {i + 1}/{n_samples} integrations completed")

            # Check timeout
            elapsed = time.perf_counter() - start_time
            if elapsed > max_time:
                print(f"\n*** TIMEOUT: Integration exceeded {max_time} seconds ***")
                print(f"Completed {i}/{n_samples} integrations before timeout")
                integration_complete = False
                completed_count = i
                break

            # Solve single trajectory using scipy.integrate.solve_ivp
            solution = solve_ivp(
                fun=ode_func,
                t_span=t_span,
                y0=ic_grid[i],
                method=method,
                t_eval=t_eval,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )

            if solution.success:
                # solution.y has shape (dof, n_steps), transpose to (n_steps, dof)
                results_list.append(solution.y.T)
                completed_count = i + 1
            else:
                print(f"Warning: Integration {i} failed - {solution.message}")
                # Add NaN array for failed integration
                results_list.append(np.full((n_steps, dof), np.nan))
                completed_count = i + 1

    except KeyboardInterrupt:
        print("\n*** Interrupted by user ***")
        integration_complete = False
    except Exception as e:
        print("\n*** ERROR during integration ***")
        print(f"Message: {e}")
        integration_complete = False

    elapsed_time = time.perf_counter() - start_time

    # Stack results if we have any
    if results_list:
        # Stack into (n_steps, n_samples, dof)
        final_trajectories = np.stack(results_list, axis=1)
        # Extract final states for display
        final_states = final_trajectories[-1]  # (n_samples, dof)
    else:
        final_trajectories = np.zeros((n_steps, 0, dof))
        final_states = np.zeros((0, dof))

    # Compute basin stability
    basin_stability_data = None
    classifications = None
    deltas = None
    verification = None

    if completed_count > 0 and final_trajectories.size > 0:
        # Classify trajectories
        classifications, deltas = classify_pendulum_trajectories(final_trajectories)
        basin_stability_data = compute_basin_stability(classifications)

        # Print basin stability results
        print_basin_stability_results(basin_stability_data, deltas)

        # Verify against reference
        verification = verify_against_reference(basin_stability_data)
        print_verification_results(verification)

    # Results
    print()
    print("=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(f"Completed integrations: {completed_count}/{n_samples}")
    if completed_count > 0:
        print(f"Time per integration: {(elapsed_time / completed_count) * 1000:.4f} ms")
    print(f"Integration status: {integration_complete}")

    if completed_count > 0 and len(final_states) > 0:
        print(f"Final state sample[0]: [{final_states[0, 0]:.6f}, {final_states[0, 1]:.6f}]")
        if completed_count > 1:
            print(f"Final state sample[-1]: [{final_states[-1, 0]:.6f}, {final_states[-1, 1]:.6f}]")
    print("=" * 50)
    print()

    # Save results
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "solver": f"scipy_{method.lower()}",
        "device": "cpu",
        "parallel": False,
        "n_samples": n_samples,
        "completed_samples": completed_count,
        "elapsed_seconds": elapsed_time,
        "time_per_integration_ms": (elapsed_time / completed_count * 1000)
        if completed_count > 0
        else None,
        "integration_complete": integration_complete,
        "rtol": rtol,
        "atol": atol,
        "max_step": max_step,
        "n_steps": n_steps,
        "basin_stability": basin_stability_data,
        "verification": verification,
        "git_commit": get_git_commit(),
    }

    return results


def save_results(results, output_dir):
    """Save benchmark results to JSON and CSV"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_file = output_dir / f"{results['solver']}_timing.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save/append CSV
    csv_file = output_dir / "all_timings.csv"

    # Create header if file doesn't exist
    if not csv_file.exists():
        with open(csv_file, "w") as f:
            f.write(
                "timestamp,solver,device,parallel,n_samples,completed_samples,"
                "elapsed_seconds,time_per_integration_ms,rtol,atol,git_commit\n"
            )

    # Append results
    with open(csv_file, "a") as f:
        time_per = (
            f"{results['time_per_integration_ms']:.6f}"
            if results["time_per_integration_ms"] is not None
            else "None"
        )
        f.write(
            f"{results['timestamp']},{results['solver']},{results['device']},"
            f"{results['parallel']},{results['n_samples']},{results['completed_samples']},"
            f"{results['elapsed_seconds']:.6f},{time_per},"
            f"{results['rtol']:.2e},{results['atol']:.2e},{results['git_commit']}\n"
        )

    print("Results saved to:")
    print(f"  - {json_file}")
    print(f"  - {csv_file}")


def main():
    """Main benchmark function"""
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "pendulum_params.json"
    with open(config_path) as f:
        config = json.load(f)

    results_dir = Path(__file__).parent.parent / "results"
    max_time = config["time_integration"]["max_integration_time_seconds"]

    # Run benchmark with primary method (DOP853)
    print("\n" + "=" * 50)
    print("Running PRIMARY method: DOP853")
    print("=" * 50 + "\n")
    results = run_benchmark(config, method="DOP853", max_time=max_time)
    save_results(results, results_dir)

    # Run benchmark with alternative method (RK45)
    print("\n" + "=" * 50)
    print("Running ALTERNATIVE method: RK45")
    print("=" * 50 + "\n")
    results_alt = run_benchmark(config, method="RK45", max_time=max_time)
    save_results(results_alt, results_dir)

    print("\nAll Scipy benchmarks complete!")


if __name__ == "__main__":
    main()
