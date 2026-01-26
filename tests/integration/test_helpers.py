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
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
from sklearn.metrics import f1_score, matthews_corrcoef  # type: ignore[reportMissingTypeStubs]

from pybasin.as_basin_stability_estimator import ASBasinStabilityEstimator
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.sampler import CsvSampler
from pybasin.study_params import SweepStudyParams, ZipStudyParams
from pybasin.types import SetupProperties

Z_THRESHOLD_OK = 2.0
Z_THRESHOLD_WARNING = 3.0


@dataclass
class StatisticalComparison:
    """Statistical comparison metrics for basin stability values.

    Used when ground truth labels are not available, only expected basin stability values.
    This is for tests like Rössler network that validate against published paper results.
    Uses two-sample z-test with significance level α=0.05.

    :ivar z_score: Z-score comparing python vs expected basin stability.
    :ivar passes_test: True if |z_score| < 1.96 (α=0.05, two-tailed test).
    """

    z_score: float
    passes_test: bool


@dataclass
class ClassificationMetrics:
    """Classification metrics comparing predicted vs ground truth labels.

    :ivar f1_per_class: F1-score for each class label.
    :ivar macro_f1: Macro-averaged F1-score across all classes.
    :ivar matthews_corrcoef: Matthews correlation coefficient for overall classification.
    """

    f1_per_class: dict[str, float]
    macro_f1: float
    matthews_corrcoef: float


def compute_statistical_comparison(
    python_bs: float, python_se: float, expected_bs: float, expected_se: float
) -> StatisticalComparison:
    """Compute statistical comparison metrics for basin stability values.

    Used when ground truth labels are not available, only expected basin stability values.
    Performs two-sample z-test with significance level α=0.05.

    :param python_bs: Basin stability computed by Python implementation.
    :param python_se: Standard error of python basin stability.
    :param expected_bs: Expected basin stability from reference (paper/MATLAB).
    :param expected_se: Standard error of expected basin stability.
    :return: StatisticalComparison with z-score and test result.
    """
    combined_se = float(np.sqrt(python_se**2 + expected_se**2))
    diff = abs(python_bs - expected_bs)

    z_score = diff / combined_se if combined_se > 0 else 0.0

    # Two-tailed z-test at α=0.05: critical value is 1.96
    passes_test = z_score < 1.96

    return StatisticalComparison(
        z_score=z_score,
        passes_test=passes_test,
    )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ClassificationMetrics:
    """Compute classification metrics between ground truth and predictions.

    Computes F1-score per class, macro-averaged F1, and Matthews correlation coefficient.

    :param y_true: Ground truth labels.
    :param y_pred: Predicted labels.
    :return: ClassificationMetrics with F1 per class, macro F1, and MCC.
    :raises ValueError: If y_true and y_pred have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, y_pred has {len(y_pred)} samples. "
            f"This indicates a bug in label extraction or sample processing."
        )

    # Get unique labels from both ground truth and predictions
    labels = sorted(set(y_true) | set(y_pred))

    # Compute F1 per class
    f1_per_class_scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    # Handle scalar or array return
    if isinstance(f1_per_class_scores, np.ndarray):
        f1_per_class = {
            str(label): float(f1) for label, f1 in zip(labels, f1_per_class_scores, strict=False)
        }
    else:
        # Single class case
        f1_per_class = {str(labels[0]): float(f1_per_class_scores)}

    # Compute macro-averaged F1
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    # Compute Matthews correlation coefficient
    mcc = float(matthews_corrcoef(y_true, y_pred))

    return ClassificationMetrics(
        f1_per_class=f1_per_class,
        macro_f1=macro_f1,
        matthews_corrcoef=mcc,
    )


@dataclass
class AttractorComparison:
    """Comparison metrics for a single attractor.

    :ivar label: Attractor label (e.g., "FP", "LC").
    :ivar python_bs: Basin stability computed by pyBasin.
    :ivar python_se: Standard error from pyBasin.
    :ivar matlab_bs: Basin stability from MATLAB bSTAB reference.
    :ivar matlab_se: Standard error from MATLAB bSTAB reference.
    :ivar f1_score: F1-score for this class label.
    :ivar matthews_corrcoef: Matthews correlation coefficient (same for all attractors).
    """

    label: str
    python_bs: float
    python_se: float
    matlab_bs: float
    matlab_se: float
    f1_score: float
    matthews_corrcoef: float

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
    f1_score: float = 0.0
    matthews_corrcoef: float = 0.0


@dataclass
class ComparisonResult:
    """Comparison result for a case study or parameter point.

    :ivar system_name: Name of the dynamical system (e.g., "pendulum", "duffing").
    :ivar case_name: Name of the case (e.g., "case1", "case2").
    :ivar attractors: List of attractor comparisons.
    :ivar parameter_value: Parameter value for parameter sweep tests (None for single-point).
    :ivar macro_f1: Macro-averaged F1-score across all classes.
    :ivar matthews_corrcoef: Matthews correlation coefficient for overall classification.
    """

    system_name: str
    case_name: str
    attractors: list[AttractorComparison]
    parameter_value: float | None = None
    macro_f1: float = 0.0
    matthews_corrcoef: float = 0.0

    def all_passed(self, f1_threshold: float = 0.9) -> bool:
        """Check if classification quality is above threshold.

        :param f1_threshold: F1-score threshold for validation.
        :return: True if macro F1-score is above threshold.
        """
        return self.macro_f1 >= f1_threshold

    def to_dict(self) -> dict[str, str | float | list[dict[str, str | float | int]] | None]:
        """Convert to dictionary for JSON serialization."""
        return {
            "system_name": self.system_name,
            "case_name": self.case_name,
            "parameter_value": self.parameter_value,
            "macro_f1": self.macro_f1,
            "matthews_corrcoef": self.matthews_corrcoef,
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
            "macro_f1": self.macro_f1,
            "matthews_corrcoef": self.matthews_corrcoef,
            "attractors": [a.to_dict() for a in self.attractors],
        }


def run_basin_stability_test(
    json_path: Path,
    setup_function: Callable[[], SetupProperties],
    label_map: dict[str, str] | None = None,
    system_name: str = "",
    case_name: str = "",
    ground_truth_csv: Path | None = None,
) -> tuple[BasinStabilityEstimator, ComparisonResult]:
    """Run basin stability test with classification metrics validation against MATLAB reference.

    This function:
    1. Loads expected results from MATLAB JSON file
    2. If ground_truth_csv is provided, uses CsvSampler with exact MATLAB ICs
    3. Verifies N matches between setup and JSON (sum of absNumMembers)
    4. Runs basin stability estimation
    5. Validates results using classification metrics (F1-score, MCC)

    :param json_path: Path to JSON file with expected results from MATLAB.
    :param setup_function: Function that returns system properties (e.g., setup_pendulum_system).
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

    # Load ground truth labels from CSV if provided
    if ground_truth_csv is not None:
        ground_truth_sampler = CsvSampler(
            ground_truth_csv,
            coordinate_columns=[f"x{i + 1}" for i in range(props["sampler"].state_dim)],
            label_column="label",
        )
        y_true = ground_truth_sampler.labels
        assert y_true is not None, "Ground truth CSV must have label column"
    else:
        raise ValueError("ground_truth_csv must be provided to compute classification metrics")

    # Get predicted labels
    if (
        bse.solution is not None
        and hasattr(bse.solution, "labels")
        and bse.solution.labels is not None
    ):
        y_pred = np.array([str(label) for label in bse.solution.labels])
    else:
        raise ValueError("Estimator solution must have labels after estimation")

    # Compute classification metrics
    metrics = compute_classification_metrics(y_true, y_pred)

    # Build comparison results with F1 per class
    attractor_comparisons: list[AttractorComparison] = []

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

        # Get F1-score for this class
        f1 = metrics.f1_per_class.get(python_label, 0.0)

        attractor_comparisons.append(
            AttractorComparison(
                label=python_label,
                python_bs=actual_bs,
                python_se=actual_std_err,
                matlab_bs=expected_bs,
                matlab_se=expected_std_err,
                f1_score=f1,
                matthews_corrcoef=metrics.matthews_corrcoef,
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
        macro_f1=metrics.macro_f1,
        matthews_corrcoef=metrics.matthews_corrcoef,
    )

    # Print classification quality summary
    print("\nClassification Metrics:")
    print(f"  Macro F1-score: {metrics.macro_f1:.4f}")
    print(f"  Matthews Correlation Coefficient: {metrics.matthews_corrcoef:.4f}")
    print("  F1 per class:")
    for label, f1 in sorted(metrics.f1_per_class.items()):
        print(f"    {label}: {f1:.4f}")

    return bse, comparison_result


def run_adaptive_basin_stability_test(
    json_path: Path,
    setup_function: Callable[[], SetupProperties],
    adaptative_parameter_name: str,
    label_keys: list[str] | None = None,
    label_map: dict[str, str] | None = None,
    system_name: str = "",
    case_name: str = "",
    ground_truths_dir: Path | None = None,
) -> tuple[ASBasinStabilityEstimator, list[ComparisonResult]]:
    """Run adaptive basin stability test with classification metrics validation against MATLAB reference.

    This function:
    1. Loads expected results from MATLAB JSON file with parameter sweep
    2. Extracts parameter values from JSON
    3. If ground_truths_dir is provided, uses CsvSampler for each parameter point with exact MATLAB ICs
    4. Creates and runs ASBasinStabilityEstimator with SweepStudyParams
    5. For each parameter point, validates results using classification metrics (F1-score, MCC)
    6. Handles JSON with either "bs_<label>" format or "bs_<label>"+"err_<label>" format

    :param json_path: Path to JSON file with expected parameter study results from MATLAB.
    :param setup_function: Function that returns system properties.
    :param adaptative_parameter_name: Name of parameter to vary.
    :param label_keys: List of label keys to check. If None, auto-detect from JSON.
    :param label_map: Optional mapping from JSON labels to Python labels.
    :param system_name: Name of the dynamical system for artifact generation.
    :param case_name: Name of the case for artifact generation.
    :param ground_truths_dir: Path to directory with parameter_index.csv and param_XXX.csv files.
        If provided, uses CsvSampler with exact MATLAB ICs for each parameter point.
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

    # Load ground truth CSVs if provided
    csv_samplers: list[CsvSampler] | None = None
    if ground_truths_dir is not None:
        # Read parameter_index.csv to map parameter values to CSV files
        index_file = ground_truths_dir / "parameter_index.csv"
        assert index_file.exists(), f"parameter_index.csv not found in {ground_truths_dir}"

        index_df = pd.read_csv(index_file)  # type: ignore

        # Get state dimension from the original sampler
        state_dim = props["sampler"].state_dim
        coordinate_columns = [f"x{i + 1}" for i in range(state_dim)]

        # Create CsvSampler for each parameter value
        # Match parameter values using tolerance for floating point comparison
        csv_samplers = []
        for param_val in parameter_values:
            # Find closest matching parameter in index
            closest_idx: int = int(
                np.argmin(
                    np.abs(
                        index_df["parameter_value"].values - param_val  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    )
                )
            )
            csv_filename: str = str(index_df.iloc[closest_idx]["filename"])  # type: ignore[reportUnknownMemberType]

            csv_file: Path = ground_truths_dir / csv_filename
            assert csv_file.exists(), f"Ground truth CSV not found: {csv_file}"
            csv_samplers.append(
                CsvSampler(csv_file, coordinate_columns=coordinate_columns, label_column="label")
            )

    # Create study params with samplers if ground truths are provided
    if csv_samplers is not None:
        study_params = ZipStudyParams(
            **{
                adaptative_parameter_name: list(parameter_values),
                "sampler": csv_samplers,
            }
        )
    else:
        study_params = SweepStudyParams(
            name=adaptative_parameter_name,
            values=list(parameter_values),
        )

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    cluster_classifier = props.get("cluster_classifier")
    assert solver is not None
    assert feature_extractor is not None
    assert cluster_classifier is not None

    # When using CSV samplers with ZipStudyParams, each sampler has its own n_samples
    # The ASBasinStabilityEstimator will use the n from each sampler in the study_params
    # For non-CSV cases, use the n from props
    n: int = props["n"] if csv_samplers is None else csv_samplers[0].n_samples

    as_bse = ASBasinStabilityEstimator(
        n=n,
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=cluster_classifier,
        study_params=study_params,
    )

    as_bse.estimate_as_bs()

    # Auto-detect label keys if not provided
    if label_keys is None:
        label_keys = [
            key.replace("bs_", "") for key in expected_results[0] if key.startswith("bs_")
        ]

    # Collect comparison results
    comparison_results: list[ComparisonResult] = []

    # Compare results at each parameter value
    for i, expected in enumerate(expected_results):
        param_value = expected["parameter"]
        actual_bs = as_bse.basin_stabilities[i]

        # Get errors for this parameter point
        errors = as_bse.get_errors(i)

        # Load ground truth labels if ground_truths_dir provided
        if ground_truths_dir is not None and csv_samplers is not None:
            y_true = csv_samplers[i].labels
            assert y_true is not None, "Ground truth CSV must have label column"

            # Get predicted labels for this parameter point from results
            result_labels_obj = as_bse.results[i]["labels"]
            if i < len(as_bse.results) and result_labels_obj is not None:
                y_pred = np.array([str(label) for label in result_labels_obj])
            else:
                raise ValueError(f"Result {i} must have labels after estimation")

            # Compute classification metrics for this parameter point
            metrics = compute_classification_metrics(y_true, y_pred)
        else:
            raise ValueError("ground_truths_dir must be provided to compute classification metrics")

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

            # Skip zero basin stability labels
            if expected_bs == 0:
                continue

            # Get actual basin stability
            actual_bs_val = actual_bs.get(python_label, 0.0)
            actual_err = errors[python_label]["e_abs"] if python_label in errors else 0.0

            # Get F1-score for this class
            f1 = metrics.f1_per_class.get(python_label, 0.0)

            attractor_comparisons.append(
                AttractorComparison(
                    label=python_label,
                    python_bs=actual_bs_val,
                    python_se=actual_err,
                    matlab_bs=expected_bs,
                    matlab_se=expected_err,
                    f1_score=f1,
                    matthews_corrcoef=metrics.matthews_corrcoef,
                )
            )

        # Build comparison result for this parameter point
        comparison_results.append(
            ComparisonResult(
                system_name=system_name,
                case_name=case_name,
                attractors=attractor_comparisons,
                parameter_value=param_value,
                macro_f1=metrics.macro_f1,
                matthews_corrcoef=metrics.matthews_corrcoef,
            )
        )

    # Print classification quality summary
    print(f"\n{'=' * 80}")
    print("Adaptive Basin Stability Classification Results")
    print(f"{'=' * 80}")
    for _i, result in enumerate(comparison_results):
        param_val = result.parameter_value
        print(f"\nParameter {param_val:.4f}:")
        print(f"  Macro F1: {result.macro_f1:.4f}")
        print(f"  MCC: {result.matthews_corrcoef:.4f}")
        print("  F1 per class:")
        for attractor in result.attractors:
            print(f"    {attractor.label}: {attractor.f1_score:.4f}")
    print(f"{'=' * 80}\n")

    return as_bse, comparison_results


def run_single_point_test(
    n: int,
    expected_bs: dict[str, float],
    setup_function: Callable[[], SetupProperties],
    expected_points: int | None = None,
) -> None:
    """Run single-point basin stability test with direct value validation.

    This function is for simple tests with one N value and no JSON reference file.
    It directly compares basin stability values without statistical testing.

    :param n: Number of initial conditions to sample.
    :param expected_bs: Expected basin stability values (label -> value).
    :param setup_function: Function that returns system properties.
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

    failures: list[str] = []
    TOLERANCE = 0.05  # Allow 5% deviation for random sampling tests
    for label, expected_value in expected_bs.items():
        actual_value = basin_stability.get(label, 0.0)
        if abs(actual_value - expected_value) > TOLERANCE:
            failures.append(
                f"Label '{label}': expected {expected_value:.4f}, got {actual_value:.4f}"
            )

    assert not failures, "Basin stability validation failed:\n" + "\n".join(failures)

    total_bs = sum(basin_stability.values())
    assert abs(total_bs - 1.0) < 0.001, f"Basin stabilities should sum to 1.0, got {total_bs}"
