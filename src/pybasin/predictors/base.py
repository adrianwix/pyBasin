from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.solution import Solution


class LabelPredictor(ABC):
    """
    Abstract base class for label prediction algorithms.

    This class provides a common interface for both supervised classifiers
    and unsupervised clusterers used in basin stability analysis.
    """

    display_name: str = "Predictor"

    @abstractmethod
    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given features.

        :param features: Feature array to predict labels for.
        :return: Array of predicted labels.
        """
        pass


class ClassifierPredictor(LabelPredictor):
    """
    Base class for supervised classifiers that require labeled template data.

    Supervised learning: Requires example trajectories with known labels to learn
    the mapping from features to basin/attractor labels. Use when attractors are known.
    """

    labels: list[str]
    template_y0: list[list[float]]  # Stored as list, converted to tensor during integration
    classifier: Any  # Type depends on subclass
    ode_params: Mapping[str, Any]

    def __init__(
        self,
        template_y0: list[list[float]],
        labels: list[str],
        ode_params: Mapping[str, Any],
        solver: SolverProtocol | None = None,
    ):
        """
        Initialize the supervised classifier.

        :param template_y0: Template initial conditions as a list of lists (e.g., [[0.5, 0.0], [2.7, 0.0]]).
                           Will be converted to tensor with appropriate device during integration.
        :param labels: Ground truth labels for template conditions.
        :param ode_params: ODE parameters mapping (dict or TypedDict with numeric values).
        :param solver: Optional solver for template integration. If provided, this solver
                       will be used instead of the main solver (useful for CPU-based
                       template integration when templates are few).
        """
        self.labels = labels
        self.template_y0 = template_y0
        self.ode_params = deepcopy(ode_params)
        self.solver = solver
        self.solution: Solution | None = None  # Populated by integrate_templates

    @property
    def has_dedicated_solver(self) -> bool:
        """Check if the classifier has its own dedicated solver for template integration."""
        return self.solver is not None

    def integrate_templates(
        self,
        solver: SolverProtocol | None,
        ode_system: ODESystemProtocol,
    ) -> None:
        """
        Integrate ODE for template initial conditions (without feature extraction).

        This method should be called before fit_with_features() to allow the main
        feature extraction to fit the scaler first.

        By default, if no dedicated solver was provided at init, this method will
        automatically create a CPU variant of the passed solver. This is because
        CPU is typically faster than GPU for small batch sizes (like templates).

        :param solver: Fallback solver if no solver was provided at init. Can be None
                       if a solver was provided during classifier initialization.
        :param ode_system: ODE system to integrate (ODESystem or JaxODESystem).
        """
        classifier_ode_system = deepcopy(ode_system)
        classifier_ode_system.params = self.ode_params

        # Determine which solver to use
        if self.solver is not None:
            # User provided a dedicated solver - use it as-is
            effective_solver = self.solver
            solver_source = "dedicated"
        elif solver is not None:
            # No dedicated solver - auto-create CPU variant for better performance
            # GPU has overhead that hurts small batch sizes (templates are typically 2-5 samples)
            if hasattr(solver, "with_device") and str(solver.device) != "cpu":
                effective_solver = solver.with_device("cpu")
                solver_source = "auto-cpu"
                print(
                    "    [ClassifierPredictor] Auto-created CPU solver for templates "
                    "(faster for small batch sizes)"
                )
            else:
                effective_solver = solver
                solver_source = "fallback"
        else:
            raise ValueError(
                "No solver available. Either pass a solver to integrate_templates() "
                "or provide one during classifier initialization."
            )

        print(f"    [ClassifierPredictor] ODE params: {classifier_ode_system.params}")
        print(f"    [ClassifierPredictor] Template ICs: {len(self.template_y0)} templates")
        print(f"    [ClassifierPredictor] Labels: {self.labels}")
        print(
            f"    [ClassifierPredictor] Using solver: {type(effective_solver).__name__} ({solver_source})"
        )

        # Convert template_y0 to tensor on the solver's device
        template_tensor = torch.tensor(
            self.template_y0, dtype=torch.float32, device=effective_solver.device
        )

        t, y = effective_solver.integrate(classifier_ode_system, template_tensor)
        self.solution = Solution(initial_condition=template_tensor, time=t, y=y)

    def fit_with_features(
        self,
        feature_extractor: FeatureExtractor,
        feature_selector: Any | None = None,
    ) -> None:
        """
        Fit the classifier using pre-integrated template solutions.

        Must call integrate_templates() first to populate self.solution.

        :param feature_extractor: Feature extractor to transform trajectories.
        :param feature_selector: Optional feature selector (already fitted on main data).
                                If provided, applies the same filtering to template features.
        :raises ValueError: If filtering removes all template features.
        """
        if self.solution is None:
            raise RuntimeError("Must call integrate_templates() before fit_with_features()")

        # Extract features from pre-integrated solution
        features = feature_extractor.extract_features(self.solution)

        # Apply feature filtering if selector provided
        if feature_selector is not None:
            features_np = features.detach().cpu().numpy()
            features_filtered_np = feature_selector.transform(features_np)

            if features_filtered_np.shape[1] == 0:
                raise ValueError(
                    f"Feature filtering removed all {features_np.shape[1]} template features. "
                    "This should not happen if the selector was fitted on main data."
                )

            features = torch.from_numpy(features_filtered_np).to(  # type: ignore[misc]
                dtype=features.dtype, device=features.device
            )

        train_x = features.detach().cpu().numpy()
        train_y = self.labels

        print(
            f"    Training classifier with {train_x.shape[0]} samples, {train_x.shape[1]} features"
        )

        self.classifier.fit(train_x, train_y)

    def fit(
        self,
        solver: SolverProtocol,
        ode_system: ODESystemProtocol,
        feature_extractor: FeatureExtractor,
    ) -> None:
        """
        Fit the classifier using template initial conditions.

        WARNING: This method extracts features from templates FIRST, which means
        the scaler will be fitted on template data (often just 2 samples). For
        better normalization, use integrate_templates() + fit_with_features()
        to allow the main data to fit the scaler first.

        :param solver: Solver to integrate the ODE system (Solver or JaxSolver).
        :param ode_system: ODE system to integrate (ODESystem or JaxODESystem).
        :param feature_extractor: Feature extractor to transform trajectories.
        """
        # Use the new two-step methods for consistency
        self.integrate_templates(solver, ode_system)
        self.fit_with_features(feature_extractor)


class ClustererPredictor(LabelPredictor):
    """
    Base class for unsupervised clustering algorithms.

    Unsupervised learning: Discovers patterns and groups in data without requiring
    labeled examples. Use when attractors/basins are unknown and need to be discovered.

    Unlike ClassifierPredictor, this class does not require template initial conditions
    or ODE parameters, as unsupervised methods work directly on features without
    needing labeled training data.
    """

    pass
