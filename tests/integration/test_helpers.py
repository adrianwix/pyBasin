"""Helper utilities for integration tests.

Test Naming Convention:
-----------------------
- test_baseline: Test with default/reference system parameters
- test_parameter_<name>: Test varying a specific system parameter (e.g., test_parameter_T, test_parameter_sigma, test_parameter_v_d)
- test_n<value>: Test with small N for validation (e.g., test_n50, test_n200)
- test_hyperparameter_<name>: Test varying a hyperparameter (e.g., test_hyperparameter_n, test_hyperparameter_rtol)

System Parameter Tests vs Hyperparameter Tests:
------------------------------------------------
System parameter tests vary dynamical system parameters (period T, sigma, velocity v_d, etc.)
and can easily use the standard utilities with z-score validation.

Hyperparameter tests vary method settings (N, solver tolerance) independent of the
dynamical system. These typically need custom validation logic (e.g., adaptive tolerance
for convergence studies) and may not fit the z-score validation pattern.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from pybasin.as_basin_stability_estimator import AdaptiveStudyParams, ASBasinStabilityEstimator
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.types import SetupProperties


def run_basin_stability_test(
    json_path: Path,
    setup_function: Callable[[], SetupProperties],
    z_threshold: float = 2.0,
    label_map: dict[str, str] | None = None,
) -> None:
    """Run basin stability test with z-score validation against MATLAB reference results.

    This function:
    1. Loads expected results from MATLAB JSON file
    2. Verifies N matches between setup and JSON (sum of absNumMembers)
    3. Runs basin stability estimation
    4. Validates results using z-score test: z = |A - B| / sqrt(SE_A^2 + SE_B^2)
    5. Asserts that differences are within z_threshold combined standard errors

    :param json_path: Path to JSON file with expected results from MATLAB.
    :param setup_function: Function that returns system properties (e.g., setup_pendulum_system).
    :param z_threshold: Z-score threshold for validation (default: 2.0, i.e., ~95% confidence).
    :param label_map: Optional mapping from JSON labels to Python labels (e.g., {"butterfly1": "chaos y_1"}).
    :raises AssertionError: If validation fails (N mismatch, z-score exceeds threshold, or label mismatch).
    """
    # Load expected results from JSON
    with open(json_path) as f:
        expected_results = json.load(f)

    # Setup system and run estimation
    props = setup_function()

    # Verify N: sum of absNumMembers should match props["n"]
    expected_n = sum(result["absNumMembers"] for result in expected_results)
    assert expected_n == props["n"], (
        f"Case study N mismatch: props['n']={props['n']} but JSON absNumMembers sum={expected_n}"
    )

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props.get("solver"),
        feature_extractor=props.get("feature_extractor"),
        predictor=props.get("cluster_classifier"),
        feature_selector=None,
    )

    basin_stability = bse.estimate_bs()

    # Verify actual N used matches expected (GridSampler may generate more points)
    if bse.y0 is not None:
        actual_n = len(bse.y0)
        print(f"\nExpected N: {expected_n}, Actual N: {actual_n}")

    # Get computed standard errors
    errors = bse.get_errors()

    # Compare results using z-score test:
    # z = |A - B| / sqrt(SE_A^2 + SE_B^2)
    # Accept if z < z_threshold (within ~z_threshold combined standard errors)
    for expected in expected_results:
        json_label = expected["label"]
        expected_bs = expected["basinStability"]
        expected_std_err = expected["standardError"]

        # Skip zero basin stability labels
        if expected_bs == 0:
            continue

        # Map JSON label to Python label if mapping provided
        python_label = (label_map.get(json_label) or json_label) if label_map else json_label

        # Get actual basin stability for this label
        actual_bs: float = basin_stability.get(python_label, 0.0)
        actual_std_err: float = errors[python_label]["e_abs"] if python_label in errors else 0.0

        # Combined standard error (both measurements have uncertainty)
        combined_std_err: float = float(np.sqrt(expected_std_err**2 + actual_std_err**2))
        difference: float = abs(actual_bs - expected_bs)

        # Special case: when both errors are 0 (basin stability = 1.0 or 0.0),
        # accept if values are exactly equal (or very close due to floating point)
        if combined_std_err == 0.0:
            assert difference < 1e-10, (
                f"Basin stability for {json_label}: expected {expected_bs:.4f}, "
                f"got {actual_bs:.4f}, difference {difference:.4f} "
                f"(deterministic case, both errors = 0)"
            )
        else:
            threshold = z_threshold * combined_std_err
            assert difference < threshold, (
                f"Basin stability for {json_label}: expected {expected_bs:.4f} ± {expected_std_err:.4f}, "
                f"got {actual_bs:.4f} ± {actual_std_err:.4f}, "
                f"difference {difference:.4f} exceeds z={z_threshold} threshold {threshold:.4f}"
            )

    # Verify we have the same labels
    expected_labels = {
        result["label"] for result in expected_results if result["basinStability"] > 0
    }
    # Apply label mapping if provided
    if label_map:
        expected_labels = {(label_map.get(label) or label) for label in expected_labels}

    actual_labels = {label for label, bs in basin_stability.items() if bs > 0}
    assert expected_labels == actual_labels, (
        f"Label mismatch: expected {expected_labels}, got {actual_labels}"
    )


def run_adaptive_basin_stability_test(
    json_path: Path,
    setup_function: Callable[[], SetupProperties],
    adaptative_parameter_name: str,
    z_threshold: float = 2.0,
    label_keys: list[str] | None = None,
    label_map: dict[str, str] | None = None,
) -> None:
    """Run adaptive basin stability test with z-score validation against MATLAB reference results.

    This function:
    1. Loads expected results from MATLAB JSON file with parameter sweep
    2. Extracts parameter values from JSON
    3. Creates and runs ASBasinStabilityEstimator
    4. For each parameter point, validates results using z-score test
    5. Handles JSON with either "bs_<label>" format or "bs_<label>"+"err_<label>" format

    :param json_path: Path to JSON file with expected parameter study results from MATLAB.
    :param setup_function: Function that returns system properties (e.g., setup_pendulum_system).
    :param adaptative_parameter_name: Name of parameter to vary (e.g., 'ode_system.params["T"]').
    :param z_threshold: Z-score threshold for validation (default: 2.0, i.e., ~95% confidence).
    :param label_keys: List of label keys to check (e.g., ["FP", "LC"]).
                       If None, auto-detect from JSON keys starting with "bs_".
    :param label_map: Optional mapping from JSON labels to Python labels (e.g., {"butterfly1": "chaos y_1"}).
    :raises AssertionError: If validation fails.
    """
    # Load expected results from JSON
    with open(json_path) as f:
        expected_results = json.load(f)

    # Setup system and run adaptive parameter study
    props = setup_function()

    # Extract parameter values from JSON
    parameter_values = np.array([result["parameter"] for result in expected_results])

    as_params = AdaptiveStudyParams(
        adaptative_parameter_values=parameter_values,
        adaptative_parameter_name=adaptative_parameter_name,
    )

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    cluster_classifier = props.get("cluster_classifier")
    assert solver is not None
    assert feature_extractor is not None
    assert cluster_classifier is not None

    as_bse = ASBasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=cluster_classifier,
        as_params=as_params,
    )

    as_bse.estimate_as_bs()

    # Auto-detect label keys if not provided
    if label_keys is None:
        label_keys = [
            key.replace("bs_", "") for key in expected_results[0] if key.startswith("bs_")
        ]

    # Collect all failures instead of stopping at first one
    failures: list[str] = []
    total_checks = 0

    # Compare results at each parameter value
    for i, expected in enumerate(expected_results):
        param_value = expected["parameter"]
        actual_bs = as_bse.basin_stabilities[i]

        # Get errors for this parameter point
        errors = as_bse.get_errors(i)

        # Check each label
        for label in label_keys:
            bs_key = f"bs_{label}"
            err_key = f"err_{label}"

            expected_bs = expected[bs_key]
            expected_err = expected.get(err_key, 0.0)  # Use 0 if no error field

            # Map JSON label to Python label if mapping provided
            python_label = (label_map.get(label) or label) if label_map else label

            # Skip if expected basin stability is 0 and it's NaN label
            if expected_bs == 0 and label == "NaN":
                actual_bs_val = actual_bs.get(python_label, 0.0)
                total_checks += 1
                if abs(actual_bs_val - expected_bs) >= 0.01:
                    failures.append(
                        f"Parameter {param_value:.4f}, {label}: "
                        f"expected {expected_bs:.4f}, got {actual_bs_val:.4f}"
                    )
                continue

            # Skip zero basin stability labels
            if expected_bs == 0:
                continue

            # Get actual basin stability
            actual_bs_val = actual_bs.get(python_label, 0.0)
            actual_err = errors[python_label]["e_abs"] if python_label in errors else 0.0

            # Combined standard error
            combined_err = float(np.sqrt(expected_err**2 + actual_err**2))
            difference = abs(actual_bs_val - expected_bs)
            total_checks += 1

            # Special case: when both errors are 0 (basin stability = 1.0 or 0.0)
            if combined_err == 0.0:
                if difference >= 1e-10:
                    failures.append(
                        f"Parameter {param_value:.4f}, {label}: "
                        f"expected {expected_bs:.4f}, got {actual_bs_val:.4f}, "
                        f"difference {difference:.4f} (deterministic case, both errors = 0)"
                    )
            else:
                threshold = z_threshold * combined_err
                if difference >= threshold:
                    z_score = difference / combined_err
                    failures.append(
                        f"Parameter {param_value:.4f}, {label}: "
                        f"expected {expected_bs:.4f} ± {expected_err:.4f}, "
                        f"got {actual_bs_val:.4f} ± {actual_err:.4f}, "
                        f"diff {difference:.4f} exceeds z={z_threshold} threshold {threshold:.4f} (z={z_score:.2f})"
                    )

    # Report results
    if failures:
        num_failures = len(failures)
        num_passed = total_checks - num_failures
        pass_rate = (num_passed / total_checks * 100) if total_checks > 0 else 0

        print(f"\n{'=' * 80}")
        print("Adaptive Basin Stability Test Results")
        print(f"{'=' * 80}")
        print(f"Total checks: {total_checks}")
        print(f"Passed: {num_passed} ({pass_rate:.1f}%)")
        print(f"Failed: {num_failures} ({100 - pass_rate:.1f}%)")
        print("\nShowing up to 5 failures:")
        print(f"{'-' * 80}")
        for failure in failures[:5]:
            print(failure)
        if num_failures > 5:
            print(f"... and {num_failures - 5} more failures")
        print(f"{'=' * 80}\n")

        raise AssertionError(f"{num_failures}/{total_checks} checks failed. See details above.")


def run_single_point_test(
    n: int,
    expected_bs: dict[str, float],
    setup_function: Callable[[], SetupProperties],
    z_threshold: float = 3.0,
    expected_points: int | None = None,
) -> None:
    """Run single-point basin stability test with inline z-score validation.

    This function is for simple tests with one N value and no JSON reference file.
    It calculates standard error as SE = sqrt(p*(1-p)/N) and validates using z-scores.

    :param n: Number of initial conditions to sample.
    :param expected_bs: Expected basin stability values (label -> value).
    :param setup_function: Function that returns system properties.
    :param z_threshold: Z-score threshold for validation (default: 3.0 for small N).
    :param expected_points: Expected number of points after sampling (for grid samplers).
    :raises AssertionError: If validation fails.
    """
    props = setup_function()

    bse = BasinStabilityEstimator(
        n=n,
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props.get("solver"),
        feature_extractor=props.get("feature_extractor"),
        predictor=props.get("cluster_classifier"),
        feature_selector=None,
    )

    basin_stability = bse.estimate_bs()

    if bse.y0 is not None:
        actual_points = len(bse.y0)
        print(f"\nActual points generated: {actual_points}")
        if expected_points is not None:
            assert actual_points == expected_points, (
                f"Expected {expected_points} points, but got {actual_points}"
            )

    actual_n = len(bse.y0) if bse.y0 is not None else n

    failures: list[str] = []
    for label, expected_value in expected_bs.items():
        actual_value = basin_stability.get(label, 0.0)
        p_hat = expected_value
        se = (p_hat * (1 - p_hat) / actual_n) ** 0.5
        z_score = abs(actual_value - expected_value) / se if se > 0 else 0

        if z_score >= z_threshold:
            failures.append(
                f"Label '{label}': expected {expected_value:.4f}, "
                f"got {actual_value:.4f}, z-score {z_score:.2f}"
            )

    assert not failures, (
        f"Basin stability validation failed (z-threshold={z_threshold}):\n" + "\n".join(failures)
    )

    total_bs = sum(basin_stability.values())
    assert abs(total_bs - 1.0) < 0.001, f"Basin stabilities should sum to 1.0, got {total_bs}"
