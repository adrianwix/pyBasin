# type: ignore
"""
Compare and visualize results from all ODE solver benchmarks
Reads timing data from results directory and generates comparison tables
"""

import json
from pathlib import Path

import pandas as pd


def load_all_results(results_dir):
    """Load all timing results from CSV file"""
    results_dir = Path(results_dir)
    csv_file = results_dir / "all_timings.csv"

    if not csv_file.exists():
        print(f"No results file found at {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    return df


def compute_speedup(df, baseline_solver="matlab_ode45", baseline_device="cpu"):
    """
    Compute speedup relative to baseline solver

    Parameters:
        df: DataFrame with timing results
        baseline_solver: Reference solver for speedup computation
        baseline_device: Reference device for baseline
    """
    # Find baseline timing
    baseline_mask = (df["solver"] == baseline_solver) & (df["device"] == baseline_device)
    baseline_results = df[baseline_mask]

    if len(baseline_results) == 0:
        print(f"Warning: No baseline results found for {baseline_solver} on {baseline_device}")
        baseline_time = None
    else:
        # Use most recent baseline result
        baseline_time = baseline_results.iloc[-1]["elapsed_seconds"]

    # Compute speedup for all entries
    if baseline_time is not None:
        df["speedup"] = baseline_time / df["elapsed_seconds"]
        df["relative_performance"] = df["elapsed_seconds"] / baseline_time * 100  # percentage
    else:
        df["speedup"] = None
        df["relative_performance"] = None

    return df, baseline_time


def generate_summary_table(df):
    """Generate summary statistics grouped by solver and device"""
    # Group by solver and device
    summary = (
        df.groupby(["solver", "device"])
        .agg(
            {
                "n_samples": "first",
                "completed_samples": "max",
                "elapsed_seconds": ["mean", "std", "min", "max"],
                "time_per_integration_ms": ["mean", "std"],
                "speedup": "mean",
                "relative_performance": "mean",
            }
        )
        .round(4)
    )

    return summary


def print_comparison_report(df, baseline_solver="matlab_ode45", baseline_device="cpu"):
    """Print a formatted comparison report"""
    print("\n" + "=" * 80)
    print("ODE SOLVER BENCHMARK COMPARISON REPORT")
    print("=" * 80)
    print(f"Baseline: {baseline_solver} on {baseline_device.upper()}")
    print("=" * 80)
    print()

    # Get unique solvers and devices
    solvers = df["solver"].unique()
    devices = df["device"].unique()

    print("Available Solvers:")
    for solver in sorted(solvers):
        solver_data = df[df["solver"] == solver]
        n_runs = len(solver_data)
        print(f"  - {solver} ({n_runs} runs)")
    print()

    print("Available Devices:")
    for device in sorted(devices):
        device_data = df[df["device"] == device]
        n_runs = len(device_data)
        print(f"  - {device.upper()} ({n_runs} runs)")
    print()

    # Summary table
    print("-" * 80)
    print("SUMMARY TABLE (Most Recent Run per Solver/Device)")
    print("-" * 80)

    # Get most recent run for each solver/device combination
    latest_results = df.sort_values("timestamp").groupby(["solver", "device"]).tail(1)

    # Sort by elapsed time
    latest_results = latest_results.sort_values("elapsed_seconds")

    # Print formatted table
    header = f"{'Solver':<25} {'Device':<8} {'Samples':<10} {'Time (s)':<12} {'ms/integ':<12} {'Speedup':<10}"
    print(header)
    print("-" * 80)

    for _, row in latest_results.iterrows():
        solver = row["solver"]
        device = row["device"]
        n_samples = int(row["completed_samples"])
        elapsed = row["elapsed_seconds"]
        time_per = row["time_per_integration_ms"]
        speedup = row["speedup"] if pd.notna(row["speedup"]) else 1.0

        # Highlight baseline
        marker = " (baseline)" if solver == baseline_solver and device == baseline_device else ""

        print(
            f"{solver:<25} {device.upper():<8} {n_samples:<10} "
            f"{elapsed:<12.4f} {time_per:<12.4f} {speedup:<10.2f}{marker}"
        )

    print("-" * 80)
    print()

    # Best performers
    print("-" * 80)
    print("TOP 5 FASTEST SOLVERS")
    print("-" * 80)

    top_5 = latest_results.nsmallest(5, "elapsed_seconds")
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        print(
            f"{idx}. {row['solver']} ({row['device'].upper()}): "
            f"{row['elapsed_seconds']:.4f}s "
            f"({row['speedup']:.2f}x speedup)"
        )

    print()

    # Device comparison
    print("-" * 80)
    print("CPU vs GPU COMPARISON")
    print("-" * 80)

    for solver in sorted(solvers):
        solver_data = latest_results[latest_results["solver"] == solver]
        cpu_data = solver_data[solver_data["device"] == "cpu"]
        gpu_data = solver_data[solver_data["device"] == "cuda"]

        if len(cpu_data) > 0 and len(gpu_data) > 0:
            cpu_time = cpu_data.iloc[0]["elapsed_seconds"]
            gpu_time = gpu_data.iloc[0]["elapsed_seconds"]
            gpu_speedup = cpu_time / gpu_time
            print(
                f"{solver}: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, "
                f"GPU speedup={gpu_speedup:.2f}x"
            )
        elif len(cpu_data) > 0:
            cpu_time = cpu_data.iloc[0]["elapsed_seconds"]
            print(f"{solver}: CPU={cpu_time:.4f}s (GPU not tested)")

    print()
    print("=" * 80)


def save_comparison_summary(df, output_dir):
    """Save comparison summary to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as CSV
    full_csv = output_dir / "comparison_full.csv"
    df.to_csv(full_csv, index=False)
    print(f"\nFull results saved to: {full_csv}")

    # Get latest results per solver/device
    latest_results = df.sort_values("timestamp").groupby(["solver", "device"]).tail(1)

    # Save summary CSV
    summary_csv = output_dir / "comparison_summary.csv"
    latest_results[
        [
            "solver",
            "device",
            "n_samples",
            "completed_samples",
            "elapsed_seconds",
            "time_per_integration_ms",
            "speedup",
            "relative_performance",
        ]
    ].to_csv(summary_csv, index=False)
    print(f"Summary results saved to: {summary_csv}")

    # Save as JSON
    summary_json = output_dir / "comparison_summary.json"
    latest_results_dict = latest_results.to_dict(orient="records")
    with open(summary_json, "w") as f:
        json.dump(latest_results_dict, f, indent=2)
    print(f"Summary JSON saved to: {summary_json}")


def main():
    """Main comparison function"""
    # Get paths
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "results"
    output_dir = results_dir

    # Load results
    print("Loading benchmark results...")
    df = load_all_results(results_dir)

    if df is None or len(df) == 0:
        print("No results found. Please run benchmarks first.")
        return

    print(f"Loaded {len(df)} benchmark results")

    # Compute speedup
    df, baseline_time = compute_speedup(df, baseline_solver="matlab_ode45", baseline_device="cpu")

    if baseline_time is not None:
        print(f"Baseline time (MATLAB ode45 on CPU): {baseline_time:.4f} seconds")

    # Generate and print report
    print_comparison_report(df, baseline_solver="matlab_ode45", baseline_device="cpu")

    # Save summary
    save_comparison_summary(df, output_dir)

    print("\nComparison complete!")


if __name__ == "__main__":
    main()
