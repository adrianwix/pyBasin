import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import matthews_corrcoef  # type: ignore[reportMissingTypeStubs]

from case_studies.duffing_oscillator.main_duffing_oscillator_with_defaults import (
    main as duffing_main,
)
from case_studies.friction.main_friction_with_defaults import main as friction_main
from case_studies.lorenz.main_lorenz_with_defaults import main as lorenz_main
from case_studies.pendulum.main_pendulum_case1_with_defaults import main as pendulum_main
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.sampler import CsvSampler

warnings.filterwarnings("ignore", message="No predictor provided")
warnings.filterwarnings("ignore", message="os.fork\\(\\) was called")


logging.getLogger("pybasin").setLevel(logging.WARNING)

TESTS_DIR = Path(__file__).parent.parent / "tests" / "integration"


def compute_mcc(
    bse: BasinStabilityEstimator,
    ground_truth_csv: Path,
    state_dim: int,
    label_column: str = "label",
) -> tuple[float, dict[str, str]]:
    """Compute MCC between predicted labels and ground truth.

    Uses majority-vote relabeling: for each predicted cluster ID, maps it to the
    ground truth label that appears most frequently in that cluster.

    :param bse: The basin stability estimator after estimation.
    :param ground_truth_csv: Path to CSV with ground truth labels.
    :param state_dim: Number of state dimensions.
    :param label_column: Name of the label column in the CSV.
    :return: Tuple of (MCC, label mapping).
    """
    gt_sampler = CsvSampler(
        ground_truth_csv,
        coordinate_columns=[f"x{i + 1}" for i in range(state_dim)],
        label_column=label_column,
    )
    y_true = gt_sampler.labels
    assert y_true is not None

    assert bse.solution is not None and bse.solution.labels is not None
    y_pred_raw = np.array([str(label) for label in bse.solution.labels])

    cluster_to_gt: dict[str, str] = {}
    for cluster_id in np.unique(y_pred_raw):
        mask: np.ndarray = y_pred_raw == cluster_id
        gt_in_cluster = y_true[mask]
        unique_labels, counts = np.unique(gt_in_cluster, return_counts=True)
        cluster_to_gt[cluster_id] = str(unique_labels[np.argmax(counts)])

    y_pred = np.array([cluster_to_gt[label] for label in y_pred_raw])

    mcc = float(matthews_corrcoef(y_true, y_pred))

    return mcc, cluster_to_gt


def print_results(
    system_name: str,
    bse: BasinStabilityEstimator,
    mcc: float,
    cluster_to_gt: dict[str, str],
    expected_json: Path,
) -> None:
    """Print basin stability results with MCC comparison.

    :param system_name: Name of the dynamical system.
    :param bse: The basin stability estimator after estimation.
    :param mcc: Matthews Correlation Coefficient.
    :param cluster_to_gt: Mapping from cluster IDs to ground truth labels.
    :param expected_json: Path to JSON with expected MATLAB results.
    """
    print(f"\n{'=' * 70}")
    print(f"{system_name} - DEFAULT SETTINGS RESULTS")
    print(f"{'=' * 70}")

    print(f"  MCC:      {mcc:.4f}")
    print(f"  Cluster -> Ground Truth mapping: {cluster_to_gt}")

    if bse.bs_vals is not None:
        errors = bse.get_errors()
        print("\n  Basin Stability (pyBasin defaults):")
        for label, bs in sorted(bse.bs_vals.items()):
            se = errors[label]["e_abs"] if label in errors else 0.0
            print(f"    {label}: {bs:.4f} ± {se:.5f}")

        with open(expected_json) as f:
            expected_results: list[dict[Any, Any]] = json.load(f)
        print("  Basin Stability (bSTAB reference):")
        for item in expected_results:
            if item["basinStability"] > 0:
                print(f"    {item['label']}: {item['basinStability']:.4f}")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    # ---- PENDULUM ----
    print("\n" + "=" * 60)
    print("PENDULUM")
    print("=" * 60)
    pendulum_csv = TESTS_DIR / "pendulum" / "ground_truths" / "case1" / "main_pendulum_case1.csv"
    pendulum_sampler = CsvSampler(pendulum_csv, coordinate_columns=["x1", "x2"])
    bse = pendulum_main(sampler_override=pendulum_sampler)
    mcc, cluster_to_gt = compute_mcc(bse, pendulum_csv, state_dim=2)
    print_results(
        "PENDULUM",
        bse,
        mcc,
        cluster_to_gt,
        TESTS_DIR / "pendulum" / "main_pendulum_case1.json",
    )

    # ---- DUFFING ----
    print("\n" + "=" * 60)
    print("DUFFING OSCILLATOR")
    print("=" * 60)
    duffing_csv = TESTS_DIR / "duffing" / "ground_truths" / "main" / "main_duffing.csv"
    duffing_sampler = CsvSampler(duffing_csv, coordinate_columns=["x1", "x2"])
    bse = duffing_main(sampler_override=duffing_sampler)
    mcc, cluster_to_gt = compute_mcc(bse, duffing_csv, state_dim=2)
    print_results(
        "DUFFING",
        bse,
        mcc,
        cluster_to_gt,
        TESTS_DIR / "duffing" / "main_duffing_supervised.json",
    )

    # ---- FRICTION ----
    print("\n" + "=" * 60)
    print("FRICTION")
    print("=" * 60)
    friction_csv = TESTS_DIR / "friction" / "ground_truths" / "main" / "main_friction.csv"
    friction_sampler = CsvSampler(friction_csv, coordinate_columns=["x1", "x2"])
    bse = friction_main(sampler_override=friction_sampler)
    mcc, cluster_to_gt = compute_mcc(bse, friction_csv, state_dim=2)
    print_results(
        "FRICTION",
        bse,
        mcc,
        cluster_to_gt,
        TESTS_DIR / "friction" / "main_friction_case1.json",
    )

    # ---- LORENZ ----
    print("\n" + "=" * 60)
    print("LORENZ")
    print("=" * 60)
    lorenz_csv = TESTS_DIR / "lorenz" / "ground_truths" / "main" / "main_lorenz.csv"
    lorenz_sampler = CsvSampler(lorenz_csv, coordinate_columns=["x1", "x2", "x3"])
    bse = lorenz_main(sampler_override=lorenz_sampler)
    mcc, cluster_to_gt = compute_mcc(bse, lorenz_csv, state_dim=3)
    print_results(
        "LORENZ",
        bse,
        mcc,
        cluster_to_gt,
        TESTS_DIR / "lorenz" / "main_lorenz.json",
    )

    print("\n" + "=" * 60)
    print("ALL CASE STUDIES WITH DEFAULTS COMPLETED")
    print("=" * 60)
