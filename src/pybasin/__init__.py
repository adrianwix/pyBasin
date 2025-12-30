"""pyBasin: Basin stability estimation for dynamical systems."""

import logging
import sys

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter
from pybasin.plotters.types import (
    FeatureSpaceOptions,
    InteractivePlotterOptions,
    PhasePlotOptions,
    StateSpaceOptions,
    TemplateSelectionOptions,
    TemplateTimeSeriesOptions,
)

# Configure library logger with default handler
_logger = logging.getLogger("pybasin")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    # _handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

__all__ = [
    "BasinStabilityEstimator",
    "FeatureSpaceOptions",
    "InteractivePlotter",
    "InteractivePlotterOptions",
    "MatplotlibPlotter",
    "PhasePlotOptions",
    "StateSpaceOptions",
    "TemplateSelectionOptions",
    "TemplateTimeSeriesOptions",
]
