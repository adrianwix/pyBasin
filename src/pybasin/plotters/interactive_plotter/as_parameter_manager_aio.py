"""AIO Parameter Manager with LRU cache for AS mode page management."""

from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import cast

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.plotters.interactive_plotter.basin_stability_aio import BasinStabilityAIO
from pybasin.plotters.interactive_plotter.bse_base_page_aio import BseBasePageAIO
from pybasin.plotters.interactive_plotter.feature_space_aio import FeatureSpaceAIO
from pybasin.plotters.interactive_plotter.state_space_aio import StateSpaceAIO
from pybasin.plotters.interactive_plotter.template_phase_plot_aio import TemplatePhasePlotAIO
from pybasin.plotters.interactive_plotter.template_time_series_aio import TemplateTimeSeriesAIO


class ASParameterManagerAIO:
    """
    AIO component managing parameter-specific page sets with LRU cache.

    Creates page instances on-demand when user navigates to a parameter value.
    Maintains LRU cache of max 5 parameter page sets. Shows loading visualization
    during page instantiation (~100ms).
    """

    MAX_CACHE_SIZE = 5

    def __init__(
        self,
        as_bse: BasinStabilityStudy,
        state_labels: dict[int, str] | None = None,
        compute_bse_callback: Callable[[int], BasinStabilityEstimator] | None = None,
    ):
        """
        Initialize AS parameter manager.

        :param as_bse: Adaptive study basin stability estimator.
        :param state_labels: Optional mapping of state indices to labels.
        :param compute_bse_callback: Optional callback to compute BSE for a parameter index.
        """
        self.as_bse = as_bse
        self.state_labels = state_labels or {}
        self.compute_bse_callback = compute_bse_callback
        self._page_cache: OrderedDict[int, dict[str, BseBasePageAIO]] = OrderedDict()

    def get_or_create_pages(self, param_index: int) -> tuple[Mapping[str, BseBasePageAIO], bool]:
        """
        Get page set for parameter index, creating if needed.

        Uses LRU cache with max 5 entries. When exceeding limit, clears entire
        cache (simplified eviction strategy).

        :param param_index: Index of parameter value.
        :return: Tuple of (page_dict, is_newly_created).
        """
        if param_index in self._page_cache:
            self._page_cache.move_to_end(param_index)
            return self._page_cache[param_index], False

        if len(self._page_cache) >= self.MAX_CACHE_SIZE:
            self._page_cache.clear()

        pages = self._create_pages_for_param(param_index)
        self._page_cache[param_index] = cast(dict[str, BseBasePageAIO], pages)
        return pages, True

    def _create_pages_for_param(self, param_index: int) -> Mapping[str, BseBasePageAIO]:
        """
        Create complete page set for specific parameter index.

        :param param_index: Index of parameter value.
        :return: Dictionary mapping page IDs to AIO page instances.
        """
        if self.compute_bse_callback is not None:
            bse = self.compute_bse_callback(param_index)
        else:
            param_value = self.as_bse.parameter_values[param_index]
            param_name = list(self.as_bse.labels[0].keys())[0] if self.as_bse.labels else "param"

            assignment = f"{param_name} = {param_value}"
            context: dict[str, object] = {
                "n": self.as_bse.n,
                "ode_system": self.as_bse.ode_system,
                "sampler": self.as_bse.sampler,
                "solver": self.as_bse.solver,
                "feature_extractor": self.as_bse.feature_extractor,
                "estimator": self.as_bse.estimator,
                "template_integrator": self.as_bse.template_integrator,
            }

            eval(compile(assignment, "<string>", "exec"), context, context)

            bse = BasinStabilityEstimator(
                n=self.as_bse.n,
                ode_system=self.as_bse.ode_system,
                sampler=self.as_bse.sampler,
                solver=self.as_bse.solver,
                feature_extractor=self.as_bse.feature_extractor,
                predictor=self.as_bse.estimator,
                template_integrator=self.as_bse.template_integrator,
                feature_selector=None,
            )
            bse.estimate_bs()

        aio_id_base = f"param-{param_index}"

        pages: dict[str, BseBasePageAIO] = {
            "state-space": StateSpaceAIO(
                bse=bse,
                aio_id=f"{aio_id_base}-state-space",
                state_labels=self.state_labels,
            ),
            "feature-space": FeatureSpaceAIO(
                bse=bse,
                aio_id=f"{aio_id_base}-feature-space",
                state_labels=self.state_labels,
            ),
            "basin-stability": BasinStabilityAIO(
                bse=bse,
                aio_id=f"{aio_id_base}-basin-stability",
                state_labels=self.state_labels,
            ),
            "templates-phase-space": TemplatePhasePlotAIO(
                bse=bse,
                aio_id=f"{aio_id_base}-phase",
                is_3d=False,
                state_labels=self.state_labels,
            ),
            "templates-time-series": TemplateTimeSeriesAIO(
                bse=bse,
                aio_id=f"{aio_id_base}-template-ts",
                state_labels=self.state_labels,
            ),
        }

        return cast(Mapping[str, BseBasePageAIO], pages)
