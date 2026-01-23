from typing import Any, NotRequired, TypedDict

import numpy as np
import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.predictors.base import LabelPredictor
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.sampler import Sampler


class ErrorInfo(TypedDict):
    """Standard error information for basin stability estimates.

    Basin stability errors are computed using Bernoulli experiment statistics:

    - e_abs = sqrt(S_B(A) * (1 - S_B(A)) / N) - absolute standard error
    - e_rel = 1 / sqrt(N * S_B(A)) - relative standard error

    :ivar e_abs: Absolute standard error of the basin stability estimate.
    :ivar e_rel: Relative standard error of the basin stability estimate.
    """

    e_abs: float
    e_rel: float


class AdaptiveStudyResult(TypedDict):
    """Results for a single parameter value in an adaptive parameter study.

    Contains complete information about basin stability estimation at one parameter value,
    including the basin stability values, error estimates, sample metadata, and optional
    detailed solution data.

    :ivar param_value: The parameter value used for this estimation, or None if no parameter is being varied.
    :ivar basin_stability: Dictionary mapping attractor labels to their basin stability values (fraction of samples).
    :ivar errors: Dictionary mapping attractor labels to their ErrorInfo (absolute and relative errors).
    :ivar n_samples: Number of initial conditions actually used (may differ from requested N due to grid rounding).
    :ivar labels: Array of attractor labels for each initial condition, or None if not available.
    :ivar bifurcation_amplitudes: Amplitude values for bifurcation analysis, or None if not computed.
    """

    param_value: float | None
    basin_stability: dict[str, float]
    errors: dict[str, ErrorInfo]
    n_samples: int
    labels: np.ndarray[Any, Any] | None
    bifurcation_amplitudes: torch.Tensor | None


class SetupProperties(TypedDict):
    """
    Standard properties returned by setup functions for case studies.

    Note: This is a flexible type definition. Actual implementations
    may use more specific types (e.g., GridSampler instead of Sampler).
    """

    n: int
    ode_system: ODESystemProtocol
    sampler: Sampler
    solver: NotRequired[SolverProtocol]
    feature_extractor: NotRequired[FeatureExtractor]
    cluster_classifier: NotRequired[LabelPredictor]
