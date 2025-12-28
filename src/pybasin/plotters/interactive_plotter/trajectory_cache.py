"""Shared trajectory cache for AIO components."""

from typing import TYPE_CHECKING

import numpy as np

from pybasin.predictors.base import ClassifierPredictor

if TYPE_CHECKING:
    from pybasin.basin_stability_estimator import BasinStabilityEstimator


class TrajectoryCache:
    """
    Centralized in-memory cache for template trajectories.

    All trajectory plot components share the same cache to avoid redundant
    expensive JAX integrations. The cache is keyed by BSE instance ID.
    """

    _cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def get_or_integrate(cls, bse: "BasinStabilityEstimator") -> tuple[np.ndarray, np.ndarray]:
        """
        Get cached trajectories or integrate and cache them.

        Uses the same integration logic as BSE.estimate_bs() to ensure consistency:
        - Uses cluster_classifier.integrate_templates() for proper ODE params and solver selection
        - Returns the integrated solution from cluster_classifier.solution

        Args:
            bse: Basin stability estimator with computed results

        Returns:
            Tuple of (time_array, trajectories_array)
            - time_array: shape (n_timesteps,)
            - trajectories_array: shape (n_timesteps, n_templates, n_states)
        """
        bse_id = id(bse)

        if bse_id in cls._cache:
            return cls._cache[bse_id]

        if not isinstance(bse.cluster_classifier, ClassifierPredictor):
            empty_time = np.array([0.0, 1.0])
            empty_traj = np.zeros((2, 1, 1))
            cls._cache[bse_id] = (empty_time, empty_traj)
            return empty_time, empty_traj

        # Check if templates are already integrated (e.g., during estimate_bs)
        if bse.cluster_classifier.solution is not None:
            # Use existing solution
            time_array = bse.cluster_classifier.solution.time.cpu().numpy()
            traj_array = bse.cluster_classifier.solution.y.cpu().numpy()
        else:
            # Integrate templates using the same logic as estimate_bs()
            # This handles ODE params and automatic CPU solver selection
            bse.cluster_classifier.integrate_templates(
                solver=bse.solver,
                ode_system=bse.ode_system,
            )

            if bse.cluster_classifier.solution is None:
                raise RuntimeError("Template integration failed to create solution")

            time_array = bse.cluster_classifier.solution.time.cpu().numpy()
            traj_array = bse.cluster_classifier.solution.y.cpu().numpy()

        cls._cache[bse_id] = (time_array, traj_array)
        return time_array, traj_array

    @classmethod
    def clear(cls, bse: "BasinStabilityEstimator | None" = None) -> None:
        """
        Clear cache for specific BSE instance or all instances.

        Args:
            bse: If provided, clear only this BSE's cache. Otherwise clear all.
        """
        if bse is not None:
            cls._cache.pop(id(bse), None)
        else:
            cls._cache.clear()
