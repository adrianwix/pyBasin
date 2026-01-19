"""Setup for the Rössler network basin stability case study."""

from typing import Any

import jax.numpy as jnp
import torch
from jax import Array

from case_studies.rossler_network.rossler_network_jax_ode import (
    RosslerNetworkJaxODE,
    RosslerNetworkParams,
)
from case_studies.rossler_network.rossler_network_topology import EDGES_I, EDGES_J, N_NODES
from case_studies.rossler_network.synchronization_classifier import (
    SynchronizationClassifier,
)
from case_studies.rossler_network.synchronization_feature_extractor import (
    SynchronizationFeatureExtractor,
)
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import JaxSolver
from pybasin.types import SetupProperties


def rossler_stop_event(t: Array, y: Array, args: Any, **kwargs: Any) -> Array:
    """
    Event function to stop integration when amplitude exceeds threshold.

    Returns positive when under threshold (continue integration),
    negative/zero when over threshold (stop integration).
    """
    max_val = 400
    max_abs_y = jnp.max(jnp.abs(y))
    return max_val - max_abs_y


def setup_rossler_network_system(k: float = 0.218) -> SetupProperties:
    """
    Setup the Rössler network system for basin stability estimation.

    Parameters
    ----------
    k : float
        Coupling strength. Must be in the stability interval (0.100, 0.336).
        Default is 0.218, which has expected S_B ≈ 0.496 from the reference paper.

    Returns
    -------
    SetupProperties
        Configuration dictionary for BasinStabilityEstimator.

    Notes
    -----
    Reference values from the paper:
        K=0.119: S_B=0.226    K=0.238: S_B=0.594
        K=0.139: S_B=0.274    K=0.258: S_B=0.628
        K=0.159: S_B=0.330    K=0.278: S_B=0.656
        K=0.179: S_B=0.346    K=0.297: S_B=0.694
        K=0.198: S_B=0.472    K=0.317: S_B=0.690
        K=0.218: S_B=0.496
    """
    n = 500

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up Rössler network system on device: {device}")
    print(f"  N = {N_NODES} nodes, k = {k}")

    params: RosslerNetworkParams = {
        "a": 0.2,
        "b": 0.2,
        "c": 7.0,
        "K": k,
        "edges_i": EDGES_I,
        "edges_j": EDGES_J,
        "N": N_NODES,
    }

    ode_system = RosslerNetworkJaxODE(params)

    min_limits = (
        [-15.0] * N_NODES  # x_i in [-15, 15]
        + [-15.0] * N_NODES  # y_i in [-15, 15]
        + [-5.0] * N_NODES  # z_i in [-5, 35]
    )
    max_limits = (
        [15.0] * N_NODES  # x_i
        + [15.0] * N_NODES  # y_i
        + [35.0] * N_NODES  # z_i
    )

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

    return {
        "n": n,
        "ode_system": ode_system,
        "sampler": sampler,
        "solver": solver,
        "feature_extractor": feature_extractor,
        "cluster_classifier": sync_classifier,
    }
