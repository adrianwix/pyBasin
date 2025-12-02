# pyright: basic
"""Modular plotter components for basin stability visualization."""

from pybasin.plotters.base_page import BasePage
from pybasin.plotters.basin_stability_page import BasinStabilityPage
from pybasin.plotters.feature_space_page import FeatureSpacePage
from pybasin.plotters.phase_plot_page import PhasePlotPage
from pybasin.plotters.state_space_page import StateSpacePage
from pybasin.plotters.template_time_series_page import TemplateTimeSeriesPage
from pybasin.plotters.utils import COLORS, WEBGL_THRESHOLD, get_color, use_webgl

__all__ = [
    "BasePage",
    "BasinStabilityPage",
    "FeatureSpacePage",
    "PhasePlotPage",
    "StateSpacePage",
    "TemplateTimeSeriesPage",
    "COLORS",
    "WEBGL_THRESHOLD",
    "get_color",
    "use_webgl",
]
