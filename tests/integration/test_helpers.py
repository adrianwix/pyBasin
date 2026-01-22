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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from pybasin.as_basin_stability_estimator import AdaptiveStudyParams, ASBasinStabilityEstimator
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.sampler import CsvSampler
from pybasin.types import SetupProperties

Z_THRESHOLD_OK = 2.0
Z_THRESHOLD_WARNING = 3.0


@dataclass
class StatisticalComparison:
    """Statistical comparison metrics between two proportion estimates.

    :ivar z_score: Z-score measuring difference in standard deviations.
    :ivar p_value: Two-tailed p-value from hypothesis test.
    :ivar ci_lower: Lower bound of 95% confidence interval for difference.
    :ivar ci_upper: Upper bound of 95% confidence interval for difference.
    :ivar confidence: Confidence level ("very_high", "high", "moderate", "low", "very_low").
    """

    z_score: float
    p_value: float
    ci_lower: float
    ci_upper: float
    confidence: str


def compute_statistical_comparison(
    value_a: float,
    se_a: float,
    value_b: float,
    se_b: float,
    alpha: float = 0.05,
) -> StatisticalComparison:
    """Compute statistical comparison between two proportion estimates.

    Uses two-sample z-test for proportions to compare estimates from independent
    Monte Carlo experiments. Returns z-score, p-value, confidence interval for
    the difference, and confidence level.

    :param value_a: First proportion estimate.
    :param se_a: Standard error of first estimate.
    :param value_b: Second proportion estimate.
    :param se_b: Standard error of second estimate.
    :param alpha: Significance level for confidence interval (default: 0.05 for 95% CI).
    :return: StatisticalComparison with z-score, p-value, CI, and confidence level.
    """
    difference = abs(value_a - value_b)
    combined_se = float(np.sqrt(se_a**2 + se_b**2))

    if combined_se > 0:
        z_score = difference / combined_se
        p_value = 2 * stats.norm.sf(abs(z_score))

        diff_signed = value_a - value_b
        z_critical = stats.norm.ppf(1 - alpha / 2)
        ci_lower = diff_signed - z_critical * combined_se
        ci_upper = diff_signed + z_critical * combined_se
    else:
        z_score = 0.0 if difference < 1e-10 else float("inf")
        p_value = 1.0 if difference < 1e-10 else 0.0
        ci_lower = 0.0
        ci_upper = 0.0

    if p_value > 0.10:
        confidence = "very_high"
    elif p_value > 0.05:
        confidence = "high"
    elif p_value > 0.01:
        confidence = "moderate"
    elif p_value > 0.001:
        confidence = "low"
    else:
        confidence = "very_low"

    return StatisticalComparison(
        z_score=z_score,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence=confidence,
    )


@dataclass
class AttractorComparison:
    """Comparison metrics for a single attractor.

    :ivar label: Attractor label (e.g., "FP", "LC").
    :ivar python_bs: Basin stability computed by pyBasin.
    :ivar python_se: Standard error from pyBasin.
    :ivar matlab_bs: Basin stability from MATLAB bSTAB reference.
    :ivar matlab_se: Standard error from MATLAB bSTAB reference.
    :ivar z_score: Z-score for the comparison.
    :ivar p_value: Two-tailed p-value from hypothesis test.
    :ivar ci_lower: Lower bound of 95% confidence interval for difference.
    :ivar ci_upper: Upper bound of 95% confidence interval for difference.
    :ivar confidence: Confidence level based on p-value.
    """

    label: str
    python_bs: float
    python_se: float
    matlab_bs: float
    matlab_se: float
    z_score: float
    p_value: float
    ci_lower: float
    ci_upper: float
    confidence: str

    @staticmethod
    def get_confidence_level(p_value: float) -> str:
        """Determine confidence level based on p-value.

        Lower p-values indicate lower confidence that implementations are equivalent.

        :param p_value: P-value from two-sample z-test.
        :return: Confidence level string.
        """
        if p_value > 0.10:
            return "very_high"
        elif p_value > 0.05:
            return "high"
        elif p_value > 0.01:
            return "moderate"
        elif p_value > 0.001:
            return "low"
        else:
            return "very_low"

    def to_dict(self) -> dict[str, str | float]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class UnsupervisedAttractorComparison(AttractorComparison):
    """Comparison metrics for a single attractor in unsupervised clustering.

    Extends AttractorComparison with cluster purity information from DBSCAN.

    :ivar dbscan_label: Original DBSCAN cluster label (numeric string).
    :ivar cluster_size: Total trajectories in this cluster.
    :ivar majority_count: Trajectories agreeing with majority template.
    :ivar purity: Fraction agreeing with majority (majority_count / cluster_size).
    """

    dbscan_label: str = ""
    cluster_size: int = 0
    majority_count: int = 0
    purity: float = 0.0


@dataclass
class ComparisonResult:
    """Comparison result for a case study or parameter point.

    :ivar system_name: Name of the dynamical system (e.g., "pendulum", "duffing").
    :ivar case_name: Name of the case (e.g., "case1", "case2").
    :ivar attractors: List of attractor comparisons.
    :ivar parameter_value: Parameter value for parameter sweep tests (None for single-point).
    :ivar z_threshold: Z-score threshold used for validation.
    """

    system_name: str
    case_name: str
    attractors: list[AttractorComparison]
    parameter_value: float | None = None
    z_threshold: float = 2.0

    def all_passed(self, z_threshold: float = 2.0) -> bool:
        """Check if all attractor comparisons passed z-threshold.

        :param z_threshold: Z-score threshold for validation.
        :return: True if all attractors have z-score below threshold.
        """
        return all(a.z_score < z_threshold for a in self.attractors)

    def to_dict(self) -> dict[str, str | float | list[dict[str, str | float | int]] | None]:
        """Convert to dictionary for JSON serialization."""
        return {
            "system_name": self.system_name,
            "case_name": self.case_name,
            "parameter_value": self.parameter_value,
            "z_threshold": self.z_threshold,
            "attractors": [a.to_dict() for a in self.attractors],
        }


@dataclass
class UnsupervisedComparisonResult(ComparisonResult):
    """Comparison result for unsupervised clustering case study.

    Extends ComparisonResult with cluster quality metrics.

    :ivar overall_agreement: Fraction of trajectories where DBSCAN matches KNN.
    :ivar adjusted_rand_index: ARI score comparing DBSCAN to KNN clustering.
    :ivar n_clusters_found: Number of clusters discovered by DBSCAN.
    :ivar n_clusters_expected: Number of clusters expected from reference.
    """

    attractors: list[UnsupervisedAttractorComparison] = None  # type: ignore[assignment]
    overall_agreement: float = 0.0
    adjusted_rand_index: float = 0.0
    n_clusters_found: int = 0
    n_clusters_expected: int = 0

    def to_dict(
        self,
    ) -> dict[str, str | float | int | list[dict[str, str | float | int]] | None]:
        """Convert to dictionary for JSON serialization."""
        return {
            "system_name": self.system_name,
            "case_name": self.case_name,
            "n_clusters_found": self.n_clusters_found,
            "n_clusters_expected": self.n_clusters_expected,
            "overall_agreement": self.overall_agreement,
            "adjusted_rand_index": self.adjusted_rand_index,
            "z_threshold": self.z_threshold,
            "attractors": [a.to_dict() for a in self.attractors],
        }


def run_basin_stability_test(
    json_path: Path,
    setup_function: Callable[[], SetupProperties],
    z_threshold: float = 2.0,
    label_map: dict[str, str] | None = None,
    system_name: str = "",
    case_name: str = "",
    ground_truth_csv: Path | None = None,
) -> tuple[BasinStabilityEstimator, ComparisonResult]:
    """Run basin stability test with z-score validation against MATLAB reference results.

    This function:
    1. Loads expected results from MATLAB JSON file
    2. If ground_truth_csv is provided, uses CsvSampler with exact MATLAB ICs
    3. Verifies N matches between setup and JSON (sum of absNumMembers)
    4. Runs basin stability estimation
    5. Validates results using z-score test: z = |A - B| / sqrt(SE_A^2 + SE_B^2)
    6. Asserts that differences are within z_threshold combined standard errors

    :param json_path: Path to JSON file with expected results from MATLAB.
    :param setup_function: Function that returns system properties (e.g., setup_pendulum_system).
    :param z_threshold: Z-score threshold for validation (default: 2.0, i.e., ~95% confidence).
    :param label_map: Optional mapping from JSON labels to Python labels.
    :param system_name: Name of the dynamical system for artifact generation.
    :param case_name: Name of the case for artifact generation.
    :param ground_truth_csv: Path to CSV with exact MATLAB initial conditions. If provided,
        uses CsvSampler instead of the sampler from setup_function.
    :return: Tuple of (BasinStabilityEstimator, ComparisonResult).
    :raises AssertionError: If validation fails.
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

    # Use CsvSampler if ground_truth_csv is provided (exact MATLAB ICs)
    if ground_truth_csv is not None:
        state_dim = props["sampler"].state_dim
        coordinate_columns = [f"x{i + 1}" for i in range(state_dim)]
        sampler = CsvSampler(ground_truth_csv, coordinate_columns=coordinate_columns)
        n_samples = sampler.n_samples
    else:
        sampler = props["sampler"]
        n_samples = props["n"]

    bse = BasinStabilityEstimator(
        n=n_samples,
        ode_system=props["ode_system"],
        sampler=sampler,
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

    # Build comparison results
    attractor_comparisons: list[AttractorComparison] = []
    failures: list[str] = []

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

        # Compute statistical comparison
        stats_comp = compute_statistical_comparison(
            actual_bs, actual_std_err, expected_bs, expected_std_err
        )

        # Check if difference is significant
        combined_std_err = float(np.sqrt(expected_std_err**2 + actual_std_err**2))
        difference = abs(actual_bs - expected_bs)

        if combined_std_err == 0.0:
            if difference >= 1e-10:
                failures.append(
                    f"Basin stability for {json_label}: expected {expected_bs:.4f}, "
                    f"got {actual_bs:.4f}, difference {difference:.4f} "
                    f"(deterministic case, both errors = 0)"
                )
        else:
            threshold = z_threshold * combined_std_err
            if difference >= threshold:
                failures.append(
                    f"Basin stability for {json_label}: expected {expected_bs:.4f} ± {expected_std_err:.4f}, "
                    f"got {actual_bs:.4f} ± {actual_std_err:.4f}, "
                    f"difference {difference:.4f} exceeds z={z_threshold} threshold {threshold:.4f} "
                    f"(p={stats_comp.p_value:.4f})"
                )

        attractor_comparisons.append(
            AttractorComparison(
                label=python_label,
                python_bs=actual_bs,
                python_se=actual_std_err,
                matlab_bs=expected_bs,
                matlab_se=expected_std_err,
                z_score=stats_comp.z_score,
                p_value=stats_comp.p_value,
                ci_lower=stats_comp.ci_lower,
                ci_upper=stats_comp.ci_upper,
                confidence=stats_comp.confidence,
            )
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

    # Build comparison result
    comparison_result = ComparisonResult(
        system_name=system_name,
        case_name=case_name,
        attractors=attractor_comparisons,
        parameter_value=None,
        z_threshold=z_threshold,
    )

    # Assert all validations passed
    assert not failures, "\n".join(failures)

    return bse, comparison_result


def run_adaptive_basin_stability_test(
    json_path: Path,
    setup_function: Callable[[], SetupProperties],
    adaptative_parameter_name: str,
    z_threshold: float = 2.0,
    label_keys: list[str] | None = None,
    label_map: dict[str, str] | None = None,
    system_name: str = "",
    case_name: str = "",
) -> tuple[ASBasinStabilityEstimator, list[ComparisonResult]]:
    """Run adaptive basin stability test with z-score validation against MATLAB reference results.

    This function:
    1. Loads expected results from MATLAB JSON file with parameter sweep
    2. Extracts parameter values from JSON
    3. Creates and runs ASBasinStabilityEstimator
    4. For each parameter point, validates results using z-score test
    5. Handles JSON with either "bs_<label>" format or "bs_<label>"+"err_<label>" format

    :param json_path: Path to JSON file with expected parameter study results from MATLAB.
    :param setup_function: Function that returns system properties.
    :param adaptative_parameter_name: Name of parameter to vary.
    :param z_threshold: Z-score threshold for validation (default: 2.0).
    :param label_keys: List of label keys to check. If None, auto-detect from JSON.
    :param label_map: Optional mapping from JSON labels to Python labels.
    :param system_name: Name of the dynamical system for artifact generation.
    :param case_name: Name of the case for artifact generation.
    :return: Tuple of (ASBasinStabilityEstimator, list of ComparisonResult per parameter).
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

    # Collect all failures and comparison results
    failures: list[str] = []
    comparison_results: list[ComparisonResult] = []
    total_checks = 0

    # Compare results at each parameter value
    for i, expected in enumerate(expected_results):
        param_value = expected["parameter"]
        actual_bs = as_bse.basin_stabilities[i]

        # Get errors for this parameter point
        errors = as_bse.get_errors(i)

        # Build attractor comparisons for this parameter point
        attractor_comparisons: list[AttractorComparison] = []

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

            # Compute statistical comparison
            stats_comp = compute_statistical_comparison(
                actual_bs_val, actual_err, expected_bs, expected_err
            )

            # Check if difference is significant
            combined_err = float(np.sqrt(expected_err**2 + actual_err**2))
            difference = abs(actual_bs_val - expected_bs)
            total_checks += 1

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
                    failures.append(
                        f"Parameter {param_value:.4f}, {label}: "
                        f"expected {expected_bs:.4f} ± {expected_err:.4f}, "
                        f"got {actual_bs_val:.4f} ± {actual_err:.4f}, "
                        f"diff {difference:.4f} exceeds z={z_threshold} threshold {threshold:.4f} "
                        f"(p={stats_comp.p_value:.4f})"
                    )

            attractor_comparisons.append(
                AttractorComparison(
                    label=python_label,
                    python_bs=actual_bs_val,
                    python_se=actual_err,
                    matlab_bs=expected_bs,
                    matlab_se=expected_err,
                    z_score=stats_comp.z_score,
                    p_value=stats_comp.p_value,
                    ci_lower=stats_comp.ci_lower,
                    ci_upper=stats_comp.ci_upper,
                    confidence=stats_comp.confidence,
                )
            )

        # Build comparison result for this parameter point
        comparison_results.append(
            ComparisonResult(
                system_name=system_name,
                case_name=case_name,
                attractors=attractor_comparisons,
                parameter_value=param_value,
                z_threshold=z_threshold,
            )
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

    return as_bse, comparison_results


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
