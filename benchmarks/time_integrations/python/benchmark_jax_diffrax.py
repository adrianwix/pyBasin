# type: ignore
"""
Benchmark JAX + Diffrax ODE solvers (Dopri5, Tsit5) for damped driven pendulum
Focuses purely on ODE integration performance (no basin stability classification)
Supports both CPU and GPU execution
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from basin_stability_utils import (
    classify_pendulum_trajectories,
    compute_basin_stability,
    print_basin_stability_results,
    print_verification_results,
    verify_against_reference,
)


def ode_pendulum(t, y, args):
    """
    Damped driven pendulum ODE system (JAX version)

    dy/dt = [y[1], -alpha*y[1] + torque - stiffness*sin(y[0])]

    Parameters in args tuple:
        alpha: dissipation coefficient
        torque: constant angular acceleration
        stiffness: stiffness coefficient (g/l)
    """
    alpha, torque, stiffness = args
    phi, omega = y
    dphi_dt = omega
    domega_dt = -alpha * omega + torque - stiffness * jnp.sin(phi)
    return jnp.array([dphi_dt, domega_dt])


def get_git_commit():
    """Get current git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_benchmark(config, method="Dopri5", device="cpu"):
    """
    Run integration benchmark for all initial conditions using JAX/Diffrax

    Parameters:
        config: Configuration dictionary
        method: Diffrax solver method ('Dopri5' or 'Tsit5')
        device: 'cpu' or 'gpu'
    """
    print("=" * 50)
    print(f"JAX + Diffrax {method} Integration Benchmark")
    print("=" * 50)
    print(f"System: {config['system']['name']}")
    print(f"Number of samples: {config['initial_conditions']['n_samples']}")
    print(
        f"Time span: [{config['time_integration']['t_start']}, {config['time_integration']['t_end']}]"
    )
    print(f"Solver: {method}")
    print(f"Device: {device}")
    print("=" * 50)
    print()

    # Set JAX device
    if device == "cuda":
        if not jax.devices("gpu"):
            print("WARNING: CUDA requested but no GPU found, falling back to CPU")
            device = "cpu"
        else:
            jax.config.update("jax_default_device", jax.devices("gpu")[0])
            print(f"Using GPU: {jax.devices('gpu')[0]}")
    else:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        print(f"Using CPU: {jax.devices('cpu')[0]}")

    # System parameters
    alpha = config["ode_parameters"]["alpha"]
    torque = config["ode_parameters"]["T"]
    stiffness = config["ode_parameters"]["K"]
    args = (alpha, torque, stiffness)

    # Time integration settings
    t0 = config["time_integration"]["t_start"]
    t1 = config["time_integration"]["t_end"]
    rtol = config["time_integration"]["rtol"]
    atol = config["time_integration"]["atol"]

    # Generate initial conditions
    n_samples = config["initial_conditions"]["n_samples"]
    roi_min = np.array(config["initial_conditions"]["roi_min"])
    roi_max = np.array(config["initial_conditions"]["roi_max"])
    seed = config["initial_conditions"]["random_seed"]

    np.random.seed(seed)
    dof = config["system"]["dof"]
    ic_grid_np = np.random.uniform(roi_min, roi_max, (n_samples, dof))

    # Convert to JAX array on target device
    ic_grid = jnp.array(ic_grid_np)

    print(f"Generated {n_samples} initial conditions (seed={seed})")
    print(f"ROI: [{roi_min[0]:.4f}, {roi_max[0]:.4f}] x [{roi_min[1]:.4f}, {roi_max[1]:.4f}]")
    print(f"Sample IC[0]: [{ic_grid[0, 0]:.4f}, {ic_grid[0, 1]:.4f}]")
    print(f"Sample IC[-1]: [{ic_grid[-1, 0]:.4f}, {ic_grid[-1, 1]:.4f}]")
    print()

    # Setup solver
    if method == "Dopri5":
        solver = diffrax.Dopri5()
    elif method == "Tsit5":
        solver = diffrax.Tsit5()
    else:
        raise ValueError(f"Unknown solver method: {method}")

    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    # Get n_steps for trajectory saving
    n_steps = config["time_integration"]["n_steps"]
    t_eval = jnp.linspace(t0, t1, n_steps, dtype=jnp.float32)
    saveat = diffrax.SaveAt(ts=t_eval)

    # Define vectorized integration function
    def integrate_single(y0):
        """Integrate a single initial condition"""
        term = diffrax.ODETerm(ode_pendulum)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=None,
            y0=y0,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=100000,
        )
        return sol.ys  # Return full trajectory

    # JIT compile and vectorize
    integrate_batch = jax.jit(jax.vmap(integrate_single))

    # Warmup run
    print("Performing warmup run...")
    warmup_ic = ic_grid[:10]
    _ = integrate_batch(warmup_ic)
    jax.block_until_ready(_)
    print("Warmup complete")
    print()

    # Run benchmark
    print(f"Starting integration (method={method}, device={device})...")
    print(f"Integrating ALL {n_samples} samples in ONE BATCH (fully vectorized)...")

    integration_complete = True
    completed_count = 0

    start_time = time.perf_counter()

    try:
        # Integrate ALL samples at once - fully vectorized!
        final_trajectories = integrate_batch(ic_grid)
        jax.block_until_ready(final_trajectories)

        completed_count = n_samples

        # Transpose from (batch, n_steps, n_dims) to (n_steps, batch, n_dims)
        final_trajectories = jnp.transpose(final_trajectories, (1, 0, 2))

        print(f"Progress: {completed_count}/{n_samples} integrations completed")

    except KeyboardInterrupt:
        print("\n*** Interrupted by user ***")
        integration_complete = False
        final_trajectories = jnp.zeros((n_steps, 0, dof))
    except Exception as e:
        print("\n*** ERROR during integration ***")
        print(f"Message: {e}")
        integration_complete = False
        final_trajectories = jnp.zeros((n_steps, 0, dof))

    elapsed_time = time.perf_counter() - start_time

    # Compute basin stability
    basin_stability_data = None
    classifications = None
    deltas = None
    verification = None

    if final_trajectories.size > 0:
        print("\nComputing basin stability...")
        # Convert JAX array to numpy for classification
        final_trajectories_np = np.array(final_trajectories)
        # final_trajectories is already (n_steps, batch, n_dims)
        classifications, deltas = classify_pendulum_trajectories(final_trajectories_np)
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

    if final_trajectories.size > 0:
        # Extract final states for display
        final_states = final_trajectories[-1]  # (n_samples, dof)
        print(f"Final state sample[0]: [{final_states[0, 0]:.6f}, {final_states[0, 1]:.6f}]")
        print(f"Final state sample[-1]: [{final_states[-1, 0]:.6f}, {final_states[-1, 1]:.6f}]")
    print("=" * 50)
    print()

    # Save results
    basin_stability_succeeded = verification["overall_pass"] if verification is not None else False

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "solver": f"jax_diffrax_{method.lower()}",
        "device": device,
        "parallel": True,  # JAX vmap is parallel
        "n_samples": n_samples,
        "completed_samples": completed_count,
        "elapsed_seconds": elapsed_time,
        "time_per_integration_ms": (elapsed_time / completed_count * 1000)
        if completed_count > 0
        else None,
        "integration_complete": integration_complete,
        "rtol": rtol,
        "atol": atol,
        "basin_stability": basin_stability_data,
        "verification": verification,
        "basin_stability_succeeded": basin_stability_succeeded,
        "git_commit": get_git_commit(),
    }

    return results


def save_results(results, output_dir):
    """Save benchmark results to JSON and CSV"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_file = output_dir / f"{results['solver']}_{results['device']}_timing.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save/append CSV
    csv_file = output_dir / "all_timings.csv"

    # Create header if file doesn't exist
    if not csv_file.exists():
        with open(csv_file, "w") as f:
            f.write(
                "timestamp,solver,device,parallel,n_samples,completed_samples,"
                "elapsed_seconds,time_per_integration_ms,rtol,atol,git_commit,basin_stability_succeeded\n"
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
            f"{results['rtol']:.2e},{results['atol']:.2e},{results['git_commit']},"
            f"{results['basin_stability_succeeded']}\n"
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

    # Detect available devices
    devices_to_test = ["cuda"]
    # if jax.devices("gpu"):
    #     devices_to_test.append("cuda")

    # Run benchmarks for each device
    for device in devices_to_test:
        # Run with primary method (Dopri5)
        print("\n" + "=" * 50)
        print(f"Running PRIMARY method: Dopri5 on {device.upper()}")
        print("=" * 50 + "\n")
        results = run_benchmark(config, method="Dopri5", device=device)
        save_results(results, results_dir)

        # Run with alternative method (Tsit5)
        # print("\n" + "=" * 50)
        # print(f"Running ALTERNATIVE method: Tsit5 on {device.upper()}")
        # print("=" * 50 + "\n")
        # results_alt = run_benchmark(config, method="Tsit5", device=device)
        # save_results(results_alt, results_dir)

    print("\nAll JAX/Diffrax benchmarks complete!")


if __name__ == "__main__":
    main()
