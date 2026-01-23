# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Two-dimensional parameter study for Rössler network basin stability.

This study investigates basin stability as a function of:
1. Coupling constant K (within stability interval)
2. Network rewiring probability p (Watts-Strogatz topology)

Implements Section 2.2.3 from the Menck et al. supplementary material.

Expected behavior:
- Basin stability increases with rewiring probability p
- Regular lattice (p=0): S_B ~ 0.30
- Small-world regime (p=0.2-0.5): S_B ~ 0.49-0.55
- Random network (p=1.0): S_B ~ 0.60

Work in progress: The code crashes when using WSL probably due to memory issues
"""

import json
import logging
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import networkx as nx
import numpy as np
import torch
from jax import Array

from case_studies.rossler_network.rossler_network_jax_ode import (
    RosslerNetworkJaxODE,
    RosslerNetworkParams,
)
from case_studies.rossler_network.synchronization_classifier import (
    SynchronizationClassifier,
)
from case_studies.rossler_network.synchronization_feature_extractor import (
    SynchronizationFeatureExtractor,
)
from pybasin.as_basin_stability_estimator import ASBasinStabilityEstimator
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import JaxSolver
from pybasin.study_params import SweepStudyParams
from pybasin.utils import generate_filename, time_execution

logger = logging.getLogger(__name__)


def build_edge_arrays_from_networkx(graph: Any) -> tuple[np.ndarray, np.ndarray]:
    """Build edge index arrays from NetworkX graph for sparse Laplacian.

    Parameters
    ----------
    graph : Any
        NetworkX graph

    Returns
    -------
    edges_i : np.ndarray
        Source node indices, shape (2*E,)
    edges_j : np.ndarray
        Target node indices, shape (2*E,)
    """
    edges_i: list[int] = []
    edges_j: list[int] = []
    for i, j in graph.edges():
        edges_i.append(int(i))
        edges_j.append(int(j))
        edges_i.append(int(j))
        edges_j.append(int(i))

    return np.array(edges_i, dtype=np.int32), np.array(edges_j, dtype=np.int32)


def compute_stability_interval(graph: Any) -> tuple[float, float]:
    """Compute stability interval for coupling constant K.

    Parameters
    ----------
    graph : Any
        NetworkX graph

    Returns
    -------
    k_min : float
        Lower bound of stability interval
    k_max : float
        Upper bound of stability interval
    """
    ALPHA_1 = 0.1232
    ALPHA_2 = 4.663

    laplacian = nx.laplacian_matrix(graph).toarray()
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.sort(eigenvalues)

    lambda_min = eigenvalues[1]
    lambda_max = eigenvalues[-1]

    k_min = ALPHA_1 / lambda_min
    k_max = ALPHA_2 / lambda_max

    return float(k_min), float(k_max)


def rossler_stop_event(t: Array, y: Array, args: Any, **kwargs: Any) -> Any:
    """Event function to stop integration when amplitude exceeds threshold."""
    MAX_VAL = 400
    max_abs_y = jnp.max(jnp.abs(y))
    return MAX_VAL - max_abs_y


def run_k_study_for_p(p: float) -> dict[str, Any]:
    """Run basin stability study for a single rewiring probability p.

    Parameters
    ----------
    p : float
        Rewiring probability for Watts-Strogatz network

    Returns
    -------
    dict[str, Any]
        Results dictionary containing p, K values, S_B values, and metadata
    """
    N_NODES = 400
    K_DEGREE = 8
    N_SAMPLES = 500
    N_K_VALUES = 11
    SEED = 42
    save_dir = f"results_2d/p_{p:.2f}"

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Running study for p = {p:.3f}")
    logger.info(f"{'=' * 80}")

    graph = nx.watts_strogatz_graph(n=N_NODES, k=K_DEGREE, p=p, seed=SEED)
    logger.info(f"Generated WS network: N={N_NODES}, k={K_DEGREE}, p={p}, seed={SEED}")
    logger.info(f"  Actual edges: {graph.number_of_edges()}")

    k_min, k_max = compute_stability_interval(graph)
    logger.info(f"Stability interval: K ∈ [{k_min:.4f}, {k_max:.4f}]")

    k_values = np.linspace(k_min, k_max, N_K_VALUES)

    edges_i, edges_j = build_edge_arrays_from_networkx(graph)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    params: RosslerNetworkParams = {
        "a": 0.2,
        "b": 0.2,
        "c": 7.0,
        "K": float(k_values[0]),
        "edges_i": jnp.array([edges_i]),
        "edges_j": jnp.array(edges_j),
        "N": N_NODES,
    }

    ode_system = RosslerNetworkJaxODE(params)

    min_limits = [-15.0] * N_NODES + [-15.0] * N_NODES + [-4.0] * N_NODES
    max_limits = [15.0] * N_NODES + [15.0] * N_NODES + [35.0] * N_NODES

    sampler = UniformRandomSampler(
        min_limits=min_limits,
        max_limits=max_limits,
        device=device,
    )

    solver = JaxSolver(
        time_span=(0, 1000),
        n_steps=1000,
        device=device,
        rtol=1e-3,
        atol=1e-6,
        use_cache=True,
        event_fn=rossler_stop_event,
    )

    feature_extractor = SynchronizationFeatureExtractor(
        n_nodes=N_NODES,
        time_steady=950,
        device=device,
    )

    sync_classifier = SynchronizationClassifier(
        epsilon=1.5,
    )

    study_params = SweepStudyParams(
        name='ode_system.params["K"]',
        values=list(k_values),
    )

    bse = ASBasinStabilityEstimator(
        n=N_SAMPLES,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=sync_classifier,
        study_params=study_params,
        save_to=save_dir,
        verbose=False,
    )

    bse.estimate_as_bs()

    sync_values: list[float] = [bs.get("synchronized", 0.0) for bs in bse.basin_stabilities]
    mean_sb = float(np.mean(sync_values))

    logger.info(f"\nResults for p = {p:.3f}:")
    logger.info(f"  Mean S_B = {mean_sb:.3f}")
    logger.info(f"  K range: [{k_values[0]:.4f}, {k_values[-1]:.4f}]")
    logger.info(f"  S_B range: [{min(sync_values):.3f}, {max(sync_values):.3f}]")

    results: dict[str, Any] = {
        "p": float(p),
        "seed": SEED,
        "n_nodes": N_NODES,
        "k_degree": K_DEGREE,
        "n_edges": graph.number_of_edges(),
        "n_samples": N_SAMPLES,
        "stability_interval": {"K_min": k_min, "K_max": k_max},
        "K_values": k_values.tolist(),
        "basin_stabilities": bse.basin_stabilities,
        "mean_sb": mean_sb,
        "parameter_values": bse.parameter_values,
    }

    return results


def main():
    """Run the two-dimensional parameter study."""
    p_values = np.arange(0.0, 1.05, 0.05)
    N_NODES = 400
    K_DEGREE = 8
    N_SAMPLES = 500
    N_K_VALUES = 11

    save_dir = Path("results_2d")
    save_dir.mkdir(exist_ok=True)

    results_2d: list[dict[str, Any]] = []

    logger.info(f"\n{'=' * 80}")
    logger.info("Two-Dimensional Parameter Study: K vs p")
    logger.info(f"{'=' * 80}")
    logger.info(f"Network: N={N_NODES}, k={K_DEGREE}")
    logger.info(f"Samples per (K,p) pair: {N_SAMPLES}")
    logger.info(f"Number of K values per p: {N_K_VALUES}")
    logger.info(f"Number of p values: {len(p_values)}")
    logger.info(f"Total parameter combinations: {len(p_values) * N_K_VALUES}")
    logger.info(f"{'=' * 80}\n")

    for p in p_values:
        results = run_k_study_for_p(float(p))
        results_2d.append(results)

    filename = generate_filename(
        name="2d_parameter_study",
        file_extension="json",
    )
    filepath = save_dir / filename
    with open(filepath, "w") as f:
        json.dump(results_2d, f, indent=2)

    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY: Basin Stability vs Rewiring Probability")
    logger.info(f"{'=' * 80}")
    logger.info(f"{'p':>6} | {'Mean S_B':>9} | {'K_min':>8} | {'K_max':>8} | {'Edges':>6}")
    logger.info("-" * 55)

    for result in results_2d:
        p = result["p"]
        mean_sb = result["mean_sb"]
        k_min = result["stability_interval"]["K_min"]
        k_max = result["stability_interval"]["K_max"]
        n_edges = result["n_edges"]
        logger.info(f"{p:>6.2f} | {mean_sb:>9.3f} | {k_min:>8.4f} | {k_max:>8.4f} | {n_edges:>6}")

    logger.info("-" * 55)
    logger.info(f"\nResults saved to: {filepath}")

    print("\n" + "=" * 80)
    print("Key Finding:")
    print("=" * 80)
    mean_sbs: list[float] = [r["mean_sb"] for r in results_2d]
    if mean_sbs[-1] > mean_sbs[0]:
        print("✓ Basin stability INCREASES with rewiring probability")
        print(f"  Regular lattice (p={p_values[0]:.1f}): S_B = {mean_sbs[0]:.3f}")
        print(f"  Random network (p={p_values[-1]:.1f}):  S_B = {mean_sbs[-1]:.3f}")
        if mean_sbs[0] > 0:
            print(f"  Relative increase: {(mean_sbs[-1] / mean_sbs[0] - 1) * 100:.1f}%")
        else:
            print(f"  Absolute increase: {mean_sbs[-1] - mean_sbs[0]:.3f}")
    else:
        print("⚠ Unexpected: Basin stability did not increase with p")
    print("=" * 80)


if __name__ == "__main__":
    time_execution("main_rossler_network_2_dimensional.py", main)
