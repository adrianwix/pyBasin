# type: ignore
"""
Benchmark Torchode ODE solver (Dopri5, Tsit5) for damped driven pendulum
Focuses purely on ODE integration performance (no basin stability classification)
Supports both CPU and GPU execution with batch integration
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchode as to
from basin_stability_utils import (
    classify_pendulum_trajectories,
    compute_basin_stability,
    print_basin_stability_results,
    print_verification_results,
    verify_against_reference,
)


def ode_pendulum(t, y, args):
    """
    Damped driven pendulum ODE system (PyTorch version)

    dy/dt = [y[1], -alpha*y[1] + torque - stiffness*sin(y[0])]

    Parameters:
        y: state tensor [batch, 2]
        args: tuple (alpha, torque, stiffness)

    Returns:
        dydt: derivatives tensor [batch, 2]
    """
    alpha, torque, stiffness = args
    phi = y[..., 0]
    omega = y[..., 1]

    dphi_dt = omega
    domega_dt = -alpha * omega + torque - stiffness * torch.sin(phi)

    return torch.stack([dphi_dt, domega_dt], dim=-1)


def get_git_commit():
    """Get current git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_benchmark(config, method="dopri5", device="cpu"):
    """
    Run integration benchmark for all initial conditions using Torchode

    Parameters:
        config: Configuration dictionary
        method: Torchode solver method ('dopri5' or 'tsit5')
        device: 'cpu' or 'cuda'
    """
    print("=" * 50)
    print(f"Torchode {method.upper()} Integration Benchmark")
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

    # Check device availability
    if device == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU")
            device = "cpu"
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    torch_device = torch.device(device)

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
    n_steps = config["time_integration"]["n_steps"]
    t_eval = torch.linspace(t0, t1, n_steps, device=torch_device)

    # Generate initial conditions
    n_samples = config["initial_conditions"]["n_samples"]
    roi_min = np.array(config["initial_conditions"]["roi_min"])
    roi_max = np.array(config["initial_conditions"]["roi_max"])
    seed = config["initial_conditions"]["random_seed"]

    np.random.seed(seed)
    dof = config["system"]["dof"]
    ic_grid_np = np.random.uniform(roi_min, roi_max, (n_samples, dof))

    # Convert to PyTorch tensor on target device
    ic_grid = torch.from_numpy(ic_grid_np).float().to(torch_device)

    print(f"Generated {n_samples} initial conditions (seed={seed})")
    print(f"ROI: [{roi_min[0]:.4f}, {roi_max[0]:.4f}] x [{roi_min[1]:.4f}, {roi_max[1]:.4f}]")
    print(f"Sample IC[0]: [{ic_grid[0, 0]:.4f}, {ic_grid[0, 1]:.4f}]")
    print(f"Sample IC[-1]: [{ic_grid[-1, 0]:.4f}, {ic_grid[-1, 1]:.4f}]")
    print()

    # Define ODE problem (capture args via closure)
    term = to.ODETerm(lambda t, y: ode_pendulum(t, y, args))

    # Select Torchode step method
    if method == "dopri5":
        step_method = to.Dopri5(term=term)
    elif method == "tsit5":
        step_method = to.Tsit5(term=term)
    else:
        raise ValueError(f"Unknown solver method: {method}")

    step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)

    def solve_batch(y0_batch: torch.Tensor) -> torch.Tensor:
        """Integrate a batch of initial conditions and return trajectories."""
        batch = y0_batch.shape[0]
        if batch == 0:
            return torch.zeros((n_steps, 0, dof), device=torch_device)

        t_start = torch.full((batch,), t_eval[0].item(), device=torch_device, dtype=t_eval.dtype)
        t_end = torch.full((batch,), t_eval[-1].item(), device=torch_device, dtype=t_eval.dtype)
        t_eval_batched = t_eval.unsqueeze(0).expand(batch, -1)

        problem = to.InitialValueProblem(
            y0=y0_batch,
            t_start=t_start,
            t_end=t_end,
            t_eval=t_eval_batched,
        )

        solution = solver.solve(problem)
        # solution.ys shape: (batch, n_steps, dof) → transpose to (n_steps, batch, dof)
        return solution.ys.transpose(0, 1)

    # Warmup run
    print("Performing warmup run...")
    warmup_ic = ic_grid[:10]

    _ = solve_batch(warmup_ic)

    if device == "cuda":
        torch.cuda.synchronize()
    print("Warmup complete")
    print()

    # Run benchmark
    print(f"Starting integration (method={method}, device={device})...")
    print(f"Integrating ALL {n_samples} samples in ONE BATCH (fully vectorized)...")

    integration_complete = True
    completed_count = 0

    start_time = time.perf_counter()

    try:
        # Integrate the entire batch at once
        final_trajectories = solve_batch(ic_grid)

        if device == "cuda":
            torch.cuda.synchronize()

        completed_count = n_samples

    except KeyboardInterrupt:
        print("\n*** Interrupted by user ***")
        integration_complete = False
        final_trajectories = torch.zeros((n_steps, 0, dof), device=torch_device)
    except Exception as e:
        print("\n*** ERROR during integration ***")
        print(f"Message: {e}")
        integration_complete = False
        final_trajectories = torch.zeros((n_steps, 0, dof), device=torch_device)

    elapsed_time = time.perf_counter() - start_time

    # Compute basin stability
    basin_stability_data = None
    classifications = None
    deltas = None
    verification = None

    if final_trajectories.numel() > 0:
        print("\nComputing basin stability...")
        final_trajectories_np = final_trajectories.detach().cpu().numpy()
        classifications, deltas = classify_pendulum_trajectories(final_trajectories_np)
        basin_stability_data = compute_basin_stability(classifications)

        print_basin_stability_results(basin_stability_data, deltas)
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

    if final_trajectories.numel() > 0:
        final_states = final_trajectories[-1]
        print(f"Final state sample[0]: [{final_states[0, 0]:.6f}, {final_states[0, 1]:.6f}]")
        print(f"Final state sample[-1]: [{final_states[-1, 0]:.6f}, {final_states[-1, 1]:.6f}]")
    print("=" * 50)
    print()

    # Save results
    basin_stability_succeeded = verification["overall_pass"] if verification is not None else False

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "solver": f"torchode_{method}",
        "device": device,
        "parallel": True,  # Batch processing is parallel
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
    devices_to_test = ["cpu"]
    if torch.cuda.is_available():
        devices_to_test.append("cuda")

    # Run benchmarks for each device
    for device in devices_to_test:
        # Run with primary method (dopri5)
        print("\n" + "=" * 50)
        print(f"Running PRIMARY method: dopri5 on {device.upper()}")
        print("=" * 50 + "\n")
        results = run_benchmark(config, method="dopri5", device=device)
        save_results(results, results_dir)

        # Run with alternative method (tsit5)
        print("\n" + "=" * 50)
        print(f"Running ALTERNATIVE method: tsit5 on {device.upper()}")
        print("=" * 50 + "\n")
        results_alt = run_benchmark(config, method="tsit5", device=device)
        save_results(results_alt, results_dir)

    print("\nAll Torchode benchmarks complete!")


if __name__ == "__main__":
    main()
