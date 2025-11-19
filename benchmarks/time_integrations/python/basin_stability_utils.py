# type: ignore
"""
Utility functions for computing basin stability from ODE integration results.

Provides classification and basin stability computation for the damped driven pendulum system.
Based on PendulumFeatureExtractor from case_studies/pendulum.
"""

import numpy as np


def classify_pendulum_trajectories(trajectories, t_star_idx=None, threshold=0.01):
    """
    Classify pendulum trajectories as Fixed Point (FP) or Limit Cycle (LC).

    Classification logic:
    - Compute delta = |max(omega) - mean(omega)| in steady-state region
    - If delta < threshold → Fixed Point (FP)
    - Otherwise → Limit Cycle (LC)

    Parameters:
        trajectories: numpy array with shape (n_steps, n_samples, n_dims)
                     where n_dims=2 (phi, omega)
        t_star_idx: Index from which to filter data (steady-state time).
                   If None, uses last 50 time points (or all if fewer than 50).
        threshold: Classification threshold (default: 0.01)

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

    # Classify: delta < threshold → FP, else → LC
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
            - std_error: standard error (Bernoulli experiment)
    """
    n_total = len(classifications)

    # Count each type
    n_fp = np.sum(classifications == "FP")
    n_lc = np.sum(classifications == "LC")

    # Compute fractions
    frac_fp = float(n_fp / n_total) if n_total > 0 else 0.0
    frac_lc = float(n_lc / n_total) if n_total > 0 else 0.0

    # Compute standard error (repeated Bernoulli experiment)
    # std_err = sqrt(p * (1 - p) / N)
    std_err_fp = np.sqrt(frac_fp * (1 - frac_fp) / n_total) if n_total > 0 else 0.0
    std_err_lc = np.sqrt(frac_lc * (1 - frac_lc) / n_total) if n_total > 0 else 0.0

    basin_stability = {
        "FP": {
            "count": int(n_fp),
            "fraction": frac_fp,
            "percentage": frac_fp * 100,
            "std_error": float(std_err_fp),
        },
        "LC": {
            "count": int(n_lc),
            "fraction": frac_lc,
            "percentage": frac_lc * 100,
            "std_error": float(std_err_lc),
        },
        "total_samples": int(n_total),
    }

    return basin_stability


def print_basin_stability_results(basin_stability, deltas=None):
    """
    Print formatted basin stability results.

    Parameters:
        basin_stability: dict from compute_basin_stability()
        deltas: optional array of delta values for statistics
    """
    print("\n" + "-" * 70)
    print("BASIN STABILITY RESULTS")
    print("-" * 70)

    for attractor in ["FP", "LC"]:
        bs = basin_stability[attractor]
        print(
            f"{attractor:3s}: {bs['count']:5d} samples ({bs['percentage']:6.2f}%) | "
            f"S = {bs['fraction']:.6f} ± {bs['std_error']:.6f}"
        )

    print(f"Total: {basin_stability['total_samples']} samples")
    print("-" * 70)

    if deltas is not None:
        print("\nDelta Statistics (|max(omega) - mean(omega)|):")
        print(f"  Min:    {deltas.min():.6f}")
        print(f"  Max:    {deltas.max():.6f}")
        print(f"  Mean:   {deltas.mean():.6f}")
        print(f"  Median: {np.median(deltas):.6f}")
        print("  Threshold: 0.01 (FP if delta < threshold)")


def get_expected_basin_stability():
    """
    Get expected basin stability values from MATLAB reference.

    For the damped driven pendulum with:
    - alpha=0.1, T=0.5, K=1.0
    - N=10000 samples
    - ROI: [-2.618, 3.665] x [-10, 10]
    - Uniform sampling with seed=42

    MATLAB Reference Results:
    - FP: 0.1520 (1520 samples) ± 0.0036
    - LC: 0.8480 (8480 samples) ± 0.0036

    Returns:
        dict with expected ranges (allowing ±3 standard errors = ±0.011)
    """
    return {
        "FP": {"min": 0.141, "max": 0.163, "typical": 0.152},
        "LC": {"min": 0.837, "max": 0.859, "typical": 0.848},
        "note": "Values based on MATLAB ode45 reference with tolerance for numerical variation",
    }


def verify_against_reference(basin_stability):
    """
    Verify basin stability against expected MATLAB reference values.

    Parameters:
        basin_stability: dict from compute_basin_stability()

    Returns:
        dict with verification status and messages
    """
    expected = get_expected_basin_stability()

    fp_frac = basin_stability["FP"]["fraction"]
    lc_frac = basin_stability["LC"]["fraction"]

    fp_ok = expected["FP"]["min"] <= fp_frac <= expected["FP"]["max"]
    lc_ok = expected["LC"]["min"] <= lc_frac <= expected["LC"]["max"]

    verification = {
        "overall_pass": fp_ok and lc_ok,
        "FP": {
            "measured": fp_frac,
            "expected_range": (expected["FP"]["min"], expected["FP"]["max"]),
            "pass": fp_ok,
        },
        "LC": {
            "measured": lc_frac,
            "expected_range": (expected["LC"]["min"], expected["LC"]["max"]),
            "pass": lc_ok,
        },
    }

    return verification


def print_verification_results(verification):
    """
    Print verification results against reference values.

    Parameters:
        verification: dict from verify_against_reference()
    """
    print("\n" + "=" * 70)
    print("VERIFICATION AGAINST MATLAB REFERENCE")
    print("=" * 70)

    for attractor in ["FP", "LC"]:
        v = verification[attractor]
        status = "✓ PASS" if v["pass"] else "✗ FAIL"
        exp_min, exp_max = v["expected_range"]
        print(
            f"{attractor}: {status} | Measured: {v['measured']:.4f} | "
            f"Expected: [{exp_min:.2f}, {exp_max:.2f}]"
        )

    print("-" * 70)
    if verification["overall_pass"]:
        print("Overall: ✓ PASS - Basin stability values are within expected range")
    else:
        print("Overall: ✗ FAIL - Basin stability values are outside expected range")
    print("=" * 70)
