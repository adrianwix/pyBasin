# pyright: basic

"""Experiment that builds nolds-based dimension & Lyapunov features for templates."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Suppress nolds warnings about autocorrelation lag and RANSAC consensus
warnings.filterwarnings(
    "ignore", message="autocorrelation declined too slowly", category=RuntimeWarning
)
warnings.filterwarnings("ignore", message="RANSAC did not reach consensus", category=RuntimeWarning)

try:
    import nolds
except ImportError as exc:
    raise ImportError("nolds is required for this experiment (pip install nolds)") from exc

if TYPE_CHECKING:
    from pybasin.solution import Solution
    from pybasin.types import SetupProperties


SetupFactory = Callable[[], "SetupProperties"]


def get_system_setups() -> list[tuple[str, SetupFactory]]:
    from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
        setup_duffing_oscillator_system,
    )
    from case_studies.friction.setup_friction_system import setup_friction_system
    from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
    from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system

    return [
        ("pendulum", setup_pendulum_system),
        ("lorenz", setup_lorenz_system),
        ("friction", setup_friction_system),
        ("duffing", setup_duffing_oscillator_system),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate nolds dimension + Lyapunov features for template trajectories before clustering."
    )
    parser.add_argument(
        "--tail-length",
        type=int,
        default=400,
        help="Number of time steps to keep for tail features.",
    )
    parser.add_argument(
        "--emb-dim", type=int, default=10, help="Embedding dimension for nolds algorithms."
    )
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps argument.")
    parser.add_argument("--min-samples", type=int, default=1, help="DBSCAN min_samples argument.")
    parser.add_argument("--no-scale", action="store_true", help="Skip scaler before clustering.")
    return parser.parse_args()


def get_tail_segment(solution: Solution, steady_time: float, tail_length: int) -> np.ndarray:
    time = solution.time.cpu().numpy()
    y = solution.y.cpu().numpy()
    start_idx = 0
    if steady_time > 0:
        start_idx = np.searchsorted(time, steady_time, side="right")
    start_idx = max(start_idx, y.shape[0] - tail_length)
    return y[start_idx:]


def safe_corr_dim(signal: np.ndarray, emb_dim: int = 10) -> float:
    if np.isnan(signal).any() or np.isinf(signal).any():
        return float("nan")
    if np.std(signal) < 1e-12:
        return 0.0
    try:
        result = nolds.corr_dim(signal, emb_dim=emb_dim)  # type: ignore[no-untyped-call]
        return float(result)  # type: ignore[arg-type]
    except (ValueError, AssertionError):
        return float("nan")


def safe_lyap_r(signal: np.ndarray, emb_dim: int = 10) -> float:
    if np.isnan(signal).any() or np.isinf(signal).any():
        return float("nan")
    if np.std(signal) < 1e-12:
        return 0.0
    try:
        result = nolds.lyap_r(signal, emb_dim=emb_dim, min_tsep=10, trajectory_len=20)  # type: ignore[no-untyped-call]
        return float(result)  # type: ignore[arg-type]
    except (ValueError, AssertionError):
        return float("nan")


def safe_lyap_e(signal: np.ndarray, emb_dim: int = 10) -> float:
    if np.isnan(signal).any() or np.isinf(signal).any():
        return float("nan")
    if np.std(signal) < 1e-12:
        return 0.0
    try:
        result = nolds.lyap_e(signal, emb_dim=emb_dim)  # type: ignore[no-untyped-call]
        return float(result[0])  # type: ignore[arg-type]  # Largest exponent
    except (ValueError, AssertionError):
        return float("nan")


def build_feature_matrix(records: list[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    if not records:
        return np.empty((0, 0)), []
    feature_names = list(records[0].keys())
    matrix = np.array([[rec[name] for name in feature_names] for rec in records], dtype=float)
    if np.isnan(matrix).any():
        medians = np.nanmedian(matrix, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        indices = np.where(np.isnan(matrix))
        matrix[indices] = medians[indices[1]]
    return matrix, feature_names


def cluster_and_report(
    system_name: str,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    labels: list[str],
    args: argparse.Namespace,
) -> None:
    print(f"\nSystem {system_name}: {len(labels)} templates, features {feature_matrix.shape[1]}")
    if feature_matrix.size == 0:
        print("  No features to cluster")
        return
    if args.no_scale:
        scaled = feature_matrix
    else:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_matrix)
    dbscan = DBSCAN(eps=args.eps, min_samples=args.min_samples)
    cluster_labels = dbscan.fit_predict(scaled)
    for idx, (label, cluster_id) in enumerate(zip(labels, cluster_labels, strict=True)):
        feats = feature_matrix[idx]
        feat_str = ", ".join(
            f"{name}={value:.3f}" for name, value in zip(feature_names, feats, strict=True)
        )
        print(f"  template {idx}: label={label}, cluster={cluster_id}, {feat_str}")
    unique_clusters = sorted({cid for cid in cluster_labels if cid != -1})
    print(f"  clusters (excluding noise): {unique_clusters}")


def inspect_system(system_name: str, setup_fn: SetupFactory, args: argparse.Namespace) -> None:
    from pybasin.cluster_classifier import SupervisedClassifier

    props = setup_fn()
    classifier = props["cluster_classifier"]
    if not isinstance(classifier, SupervisedClassifier):
        print(f"  Skipping {system_name}: classifier is not SupervisedClassifier")
        return
    solver = props["solver"]
    ode_system = props["ode_system"]
    feature_extractor = props["feature_extractor"]
    classifier.integrate_templates(solver, ode_system)
    solution = classifier.solution
    if solution is None:
        raise RuntimeError("Classifier failed to integrate templates")
    tail = get_tail_segment(solution, feature_extractor.time_steady, args.tail_length)
    records: list[dict[str, float]] = []
    for sample_idx in range(len(classifier.labels)):
        sample = tail[:, sample_idx, :]
        record = {}
        for state_idx in range(sample.shape[1]):
            signal = sample[:, state_idx]
            record[f"state{state_idx}_corr_dim"] = safe_corr_dim(signal, emb_dim=args.emb_dim)
            record[f"state{state_idx}_lyap_r"] = safe_lyap_r(signal, emb_dim=args.emb_dim)
            record[f"state{state_idx}_lyap_e"] = safe_lyap_e(signal, emb_dim=args.emb_dim)
        records.append(record)
    matrix, feature_names = build_feature_matrix(records)
    cluster_and_report(system_name, matrix, feature_names, classifier.labels, args)


def main() -> None:
    args = parse_args()
    for system_name, setup_fn in get_system_setups():
        inspect_system(system_name, setup_fn, args)


if __name__ == "__main__":
    main()
