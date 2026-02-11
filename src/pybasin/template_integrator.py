"""Template integrator for supervised classifiers in basin stability analysis."""

import logging
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.solution import Solution

logger = logging.getLogger(__name__)


class TemplateIntegrator:
    """Integrates template initial conditions and extracts training data for supervised classifiers.

    Handles the ODE integration of known attractor templates and feature extraction.

    :ivar template_y0: Template initial conditions.
    :ivar labels: Ground truth labels for each template.
    :ivar ode_params: ODE parameters for template integration (may differ from main study).
    :ivar solver: Optional dedicated solver for template integration.
    :ivar solution: Populated after :meth:`integrate` is called.
    """

    def __init__(
        self,
        template_y0: list[list[float]],
        labels: list[str],
        ode_params: Mapping[str, Any],
        solver: SolverProtocol | None = None,
    ):
        """Initialize the template integrator.

        :param template_y0: Template initial conditions as a list of lists
            (e.g., ``[[0.5, 0.0], [2.7, 0.0]]``). Will be converted to tensor
            with appropriate device during integration.
        :param labels: Ground truth labels for template conditions.
        :param ode_params: ODE parameters mapping (dict or TypedDict with numeric values).
        :param solver: Optional dedicated solver for template integration. If provided,
            this solver will be used instead of the main solver (useful for CPU-based
            template integration when templates are few).
        """
        self.template_y0 = template_y0
        self.labels = labels
        self.ode_params = deepcopy(ode_params)
        self.solver = solver
        self.solution: Solution | None = None

    @property
    def has_dedicated_solver(self) -> bool:
        """Check if a dedicated solver was provided for template integration."""
        return self.solver is not None

    def integrate(
        self,
        solver: SolverProtocol | None,
        ode_system: ODESystemProtocol,
    ) -> None:
        """Integrate ODE for template initial conditions.

        If no dedicated solver was provided at init, automatically creates a CPU
        variant of the passed solver for better performance with small batch sizes.

        :param solver: Fallback solver if none was provided at init. Can be None
            if a solver was provided during initialization.
        :param ode_system: ODE system to integrate.
        :raises ValueError: If no solver is available.
        """
        template_ode_system = deepcopy(ode_system)
        template_ode_system.params = self.ode_params

        if self.solver is not None:
            effective_solver = self.solver
        elif solver is not None:
            if hasattr(solver, "with_device") and str(solver.device) != "cpu":
                effective_solver = solver.with_device("cpu")
                logger.info(
                    "[TemplateIntegrator] Auto-created CPU solver (faster for small batch sizes)"
                )
            else:
                effective_solver = solver
        else:
            raise ValueError(
                "No solver available. Either pass a solver to integrate() "
                "or provide one during initialization."
            )

        logger.info("[TemplateIntegrator] ODE params: %s", template_ode_system.params)
        logger.info("[TemplateIntegrator] Template ICs: %d templates", len(self.template_y0))
        logger.info("[TemplateIntegrator] Labels: %s", self.labels)
        logger.info("[TemplateIntegrator] Using solver: %s", type(effective_solver).__name__)

        template_tensor = torch.tensor(
            self.template_y0, dtype=torch.float32, device=effective_solver.device
        )

        t, y = effective_solver.integrate(template_ode_system, template_tensor)
        self.solution = Solution(initial_condition=template_tensor, time=t, y=y)

    def get_training_data(
        self,
        feature_extractor: FeatureExtractor,
        feature_selector: Any | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Extract features from integrated templates and return training data.

        Must call :meth:`integrate` first to populate ``self.solution``.

        :param feature_extractor: Feature extractor (already fitted on main data).
        :param feature_selector: Optional feature selector (already fitted on main data).
            If provided, applies the same filtering to template features.
        :return: Tuple of ``(X_train, y_labels)`` ready for ``classifier.fit()``.
        :raises RuntimeError: If :meth:`integrate` was not called first.
        :raises ValueError: If filtering removes all template features.
        """
        if self.solution is None:
            raise RuntimeError("Must call integrate() before get_training_data()")

        features = feature_extractor.extract_features(self.solution)

        if feature_selector is not None:
            features_np = features.detach().cpu().numpy()
            features_filtered_np = feature_selector.transform(features_np)

            if features_filtered_np.shape[1] == 0:
                raise ValueError(
                    f"Feature filtering removed all {features_np.shape[1]} template features. "
                    "This should not happen if the selector was fitted on main data."
                )

            train_x = features_filtered_np
        else:
            train_x = features.detach().cpu().numpy()

        logger.info(
            "Template training data: %d samples, %d features",
            train_x.shape[0],
            train_x.shape[1],
        )

        return train_x, self.labels
