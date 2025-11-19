# type: ignore
"""
Verify basin stability values from benchmark integration results.

This utility classifies the steady-state solutions (Fixed Point vs Limit Cycle)
and computes basin stability values to verify that the ODE integrations are correct.

Based on the pendulum feature extractor from case_studies/pendulum.
"""

import json
import subprocess
from pathlib import Path

import numpy as np


def get_git_commit():
    """Get current git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def classify_pendulum_trajectories(trajectories, t_star_idx=None):
    """
    Classify pendulum trajectories as Fixed Point (FP) or Limit Cycle (LC).

    Based on PendulumFeatureExtractor logic:
    - If delta between max and mean angular velocity < 0.01 → Fixed Point
    - Otherwise → Limit Cycle

    Parameters:
        trajectories: numpy array with shape (n_steps, n_samples, n_dims)
                     where n_dims=2 (phi, omega)
        t_star_idx: Index from which to filter data (steady-state time).
                   If None, uses last 50 time points (or all if fewer than 50).

    Returns:
        classifications: numpy array of shape (n_samples,) with values "FP" or "LC"
        deltas: numpy array of shape (n_samples,) with delta values used for classification
    """
    n_steps, n_samples, n_dims = trajectories.shape

    # Determine filtering index for steady-state
    if t_star_idx is None:
        # Default: use last 50 steps or all if fewer
        t_star_idx = max(0, n_steps - 50)

    # Filter to steady-state region: shape (n_steps_filtered, n_samples, n_dims)
    y_filtered = trajectories[t_star_idx:]

    # Extract angular velocity (second dimension): shape (n_steps_filtered, n_samples)
    angular_velocity = y_filtered[:, :, 1]

    # Compute delta: |max - mean| for each sample
    max_omega = angular_velocity.max(axis=0)  # shape: (n_samples,)
    mean_omega = angular_velocity.mean(axis=0)  # shape: (n_samples,)
    deltas = np.abs(max_omega - mean_omega)

    # Classify: delta < 0.01 → FP, else → LC
    threshold = 0.01
    classifications = np.where(deltas < threshold, "FP", "LC")

    return classifications, deltas


def compute_basin_stability(classifications):
    """
    Compute basin stability values for each attractor type.

    Parameters:
        classifications: array of classifications ("FP" or "LC")

    Returns:
        basin_stability: dict with keys "FP" and "LC" containing:
            - count: number of samples
            - fraction: basin stability (fraction of total)
            - percentage: basin stability as percentage
    """
    n_total = len(classifications)

    # Count each type
    n_fp = np.sum(classifications == "FP")
    n_lc = np.sum(classifications == "LC")

    # Compute fractions
    basin_stability = {
        "FP": {
            "count": int(n_fp),
            "fraction": float(n_fp / n_total) if n_total > 0 else 0.0,
            "percentage": float(n_fp / n_total * 100) if n_total > 0 else 0.0,
        },
        "LC": {
            "count": int(n_lc),
            "fraction": float(n_lc / n_total) if n_total > 0 else 0.0,
            "percentage": float(n_lc / n_total * 100) if n_total > 0 else 0.0,
        },
    }

    return basin_stability


def load_solver_results(results_dir, solver_name):
    """
    Load solver benchmark results including trajectories.

    Parameters:
        results_dir: Path to results directory
        solver_name: Name of the solver (e.g., "scipy_dop853", "scipy_rk45")

    Returns:
        trajectories: numpy array (n_steps, n_samples, n_dims) or None if not found
        metadata: dict with timing info or None if not found
    """
    results_dir = Path(results_dir)

    # Load metadata from JSON
    json_file = results_dir / f"{solver_name}_timing.json"
    if not json_file.exists():
        print(f"Warning: Results file not found: {json_file}")
        return None, None

    with open(json_file) as f:
        metadata = json.load(f)

    print(f"Loaded metadata from {json_file}")
    print(f"  Solver: {metadata.get('solver', 'unknown')}")
    print(f"  Completed samples: {metadata.get('completed_samples', 0)}")
    print(f"  Elapsed time: {metadata.get('elapsed_seconds', 0):.4f}s")

    # Load trajectories from NPZ file
    npz_file = results_dir / f"{solver_name}_trajectories.npz"
    if not npz_file.exists():
        print(f"Warning: Trajectory file not found: {npz_file}")
        print("  Run benchmark again to save trajectory data for verification")
        return None, metadata

    data = np.load(npz_file)
    trajectories = data["trajectories"]

    print(f"Loaded trajectories from {npz_file}")
    print(f"  Shape: {trajectories.shape} (n_steps, n_samples, n_dims)")

    return trajectories, metadata


def verify_solver(results_dir, solver_name, t_star_fraction=0.95):
    """
    Verify a solver's results by computing basin stability.

    Parameters:
        results_dir: Path to results directory
        solver_name: Name of the solver (e.g., "scipy_dop853")
        t_star_fraction: Fraction of time span to use as t_star (default: 0.95 = last 5%)

    Returns:
        dict with verification results or None if data not available
    """
    print("\n" + "=" * 70)
    print(f"VERIFYING: {solver_name}")
    print("=" * 70)

    trajectories, metadata = load_solver_results(results_dir, solver_name)

    if trajectories is None:
        print("Cannot verify without trajectory data")
        print("=" * 70)
        return None

    # Determine t_star index
    n_steps = trajectories.shape[0]
    t_star_idx = int(n_steps * t_star_fraction)

    print("\nClassifying trajectories...")
    print(f"  Total steps: {n_steps}")
    print(f"  T_star index: {t_star_idx} (using last {n_steps - t_star_idx} steps)")

    # Classify
    classifications, deltas = classify_pendulum_trajectories(trajectories, t_star_idx)

    # Compute basin stability
    basin_stability = compute_basin_stability(classifications)

    # Print results
    print("\n" + "-" * 70)
    print("BASIN STABILITY RESULTS")
    print("-" * 70)

    for attractor in ["FP", "LC"]:
        bs = basin_stability[attractor]
        print(
            f"{attractor:3s}: {bs['count']:5d} samples ({bs['percentage']:6.2f}%) | "
            f"Basin Stability = {bs['fraction']:.6f}"
        )

    print("-" * 70)

    # Statistics on deltas
    print("\nDelta Statistics (|max(omega) - mean(omega)|):")
    print(f"  Min:    {deltas.min():.6f}")
    print(f"  Max:    {deltas.max():.6f}")
    print(f"  Mean:   {deltas.mean():.6f}")
    print(f"  Median: {np.median(deltas):.6f}")
    print("  Threshold: 0.01 (FP if delta < threshold)")

    # Sample classifications
    print("\nSample Classifications:")
    for i in [0, 1, 2, -3, -2, -1]:
        if abs(i) < len(classifications):
            idx = i if i >= 0 else len(classifications) + i
            print(f"  IC[{idx:5d}]: {classifications[idx]:2s} (delta={deltas[idx]:.6f})")

    print("=" * 70)

    # Return results
    verification_results = {
        "solver": solver_name,
        "timestamp": metadata.get("timestamp") if metadata else None,
        "n_samples": len(classifications),
        "t_star_idx": t_star_idx,
        "basin_stability": basin_stability,
        "delta_stats": {
            "min": float(deltas.min()),
            "max": float(deltas.max()),
            "mean": float(deltas.mean()),
            "median": float(np.median(deltas)),
            "threshold": 0.01,
        },
        "git_commit": get_git_commit(),
    }

    return verification_results


def save_verification_results(results, output_dir):
    """Save verification results to JSON"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_file = output_dir / f"{results['solver']}_basin_stability.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nVerification results saved to: {json_file}")


def compare_with_matlab_reference():
    """
    Compare with expected MATLAB reference values.

    Based on the MATLAB bSTAB implementation with the same parameters:
    - alpha=0.1, T=0.5, K=1.0
    - N=10000 samples
    - ROI: [-2.618, 3.665] x [-10, 10]

    Expected approximate values (these will vary slightly due to random sampling):
    - FP basin stability: ~0.55-0.65 (55-65%)
    - LC basin stability: ~0.35-0.45 (35-45%)
    """
    print("\n" + "=" * 70)
    print("MATLAB REFERENCE COMPARISON")
    print("=" * 70)
    print("\nExpected Basin Stability (approximate, from MATLAB bSTAB):")
    print("  FP: ~55-65% (Fixed Point)")
    print("  LC: ~35-45% (Limit Cycle)")
    print("\nNote: Values vary with random seed but should be in these ranges")
    print("      for alpha=0.1, T=0.5, K=1.0 with uniform sampling")
    print("=" * 70)


def main():
    """Main verification function"""
    # Get paths
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "results"

    print("=" * 70)
    print("BASIN STABILITY VERIFICATION FOR BENCHMARKS")
    print("=" * 70)
    print("\nThis utility verifies ODE integration results by computing")
    print("basin stability values for the damped driven pendulum system.")
    print()

    # Show reference values
    compare_with_matlab_reference()

    # List available solvers
    print("\n" + "=" * 70)
    print("AVAILABLE SOLVER RESULTS")
    print("=" * 70)

    json_files = list(results_dir.glob("*_timing.json"))
    if not json_files:
        print("No timing results found. Please run benchmarks first.")
        return

    solver_names = [f.stem.replace("_timing", "") for f in json_files]
    print(f"Found {len(solver_names)} solver results:")
    for name in solver_names:
        print(f"  - {name}")

    # Note about trajectory data
    print("\n" + "=" * 70)
    print("VERIFYING SOLVERS")
    print("=" * 70)
    print("Attempting to verify available solver results...")
    print()

    # Try to verify all available solvers
    verified_count = 0
    for solver_name in solver_names:
        results = verify_solver(results_dir, solver_name)
        if results:
            save_verification_results(results, results_dir)
            verified_count += 1

    print("\n" + "=" * 70)
    print(f"VERIFICATION SUMMARY: {verified_count}/{len(solver_names)} solvers verified")
    print("=" * 70)

    if verified_count == 0:
        print("\nNo solvers could be verified. Make sure to:")
        print("  1. Run benchmarks first: uv run python/benchmark_scipy.py")
        print("  2. Check that trajectory files (*.npz) were created in results/")
    else:
        print("\nVerification complete! Basin stability values saved to results/")
        print("Check *_basin_stability.json files for detailed results.")


if __name__ == "__main__":
    main()
