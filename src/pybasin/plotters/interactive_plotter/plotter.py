# pyright: basic
"""Interactive web-based plotter using Dash and Plotly with Mantine components."""

import logging
from typing import Literal

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
from dash import ALL, Dash, Input, NoUpdate, Output, State, ctx, dcc, html, no_update
from dash.development.base_component import Component

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.plotters.types import InteractivePlotterOptions

from .as_parameter_manager_aio import ASParameterManagerAIO
from .basin_stability_aio import BasinStabilityAIO
from .feature_space_aio import FeatureSpaceAIO
from .ids import IDs
from .param_bifurcation_aio import ParamBifurcationAIO
from .param_overview_aio import ParamOverviewAIO
from .state_space_aio import StateSpaceAIO
from .template_phase_plot_aio import TemplatePhasePlotAIO
from .template_time_series_aio import TemplateTimeSeriesAIO
from .trajectory_modal_aio import TrajectoryModalAIO

logger = logging.getLogger(__name__)


NavActiveState = Literal["exact", "partial"] | None


class InteractivePlotter:
    """
    Interactive web-based plotter for basin stability visualization.

    Uses Dash with Mantine components for a modern UI and Plotly for
    interactive visualizations. Each page owns its controls, plot, and callbacks.

    :ivar bse: BasinStabilityEstimator instance with computed results.
    :ivar state_labels: Optional mapping of state indices to custom labels.
    :ivar app: Dash application instance.
    """

    # Main layout IDs
    PAGE_CONTAINER = "page-container"
    CURRENT_VIEW = "current-view"
    URL = "url"

    def __init__(
        self,
        bse: BasinStabilityEstimator | BasinStabilityStudy,
        state_labels: dict[int, str] | None = None,
        options: InteractivePlotterOptions | None = None,
    ):
        """
        Initialize the Plotter.

        :param bse: BasinStabilityEstimator or BasinStabilityStudy instance.
        :param state_labels: Optional dict mapping state indices to labels,
                            e.g., {0: "θ", 1: "ω"} for a pendulum system.
        :param options: Optional configuration for default control values.
        """
        self.is_adaptive_study = isinstance(bse, BasinStabilityStudy)

        if isinstance(bse, BasinStabilityStudy):
            self.as_bse: BasinStabilityStudy = bse
        else:
            self.bse: BasinStabilityEstimator = bse

        self.state_labels = state_labels or {}
        self.options = options or InteractivePlotterOptions()

        # Override initial view for AS mode if not explicitly set
        if self.is_adaptive_study and options is None:
            self.options.initial_view = IDs.PARAM_OVERVIEW

        self.app: Dash | None = None

        if self.is_adaptive_study:
            self._validate_as_bse()
            self._init_as_pages()
        else:
            self._validate_bse()
            self._init_bse_pages()

    def _validate_bse(self) -> None:
        """Validate that BSE has required data for plotting."""
        if self.bse.solution is None:
            raise ValueError("No solution available. Run estimate_bs() first.")
        if self.bse.y0 is None:
            raise ValueError("No initial conditions available. Run estimate_bs() first.")
        if self.bse.solution.labels is None:
            raise ValueError("No labels available. Run estimate_bs() first.")
        if self.bse.bs_vals is None:
            raise ValueError("No basin stability values available. Run estimate_bs() first.")

    def _validate_as_bse(self) -> None:
        """Validate that AS-BSE has required data for plotting."""
        if len(self.as_bse.parameter_values) == 0:
            raise ValueError("No parameter values available. Run estimate_as_bs() first.")
        if len(self.as_bse.basin_stabilities) != len(self.as_bse.parameter_values):
            raise ValueError("Basin stabilities length doesn't match parameter values length.")
        if len(self.as_bse.results) != len(self.as_bse.parameter_values):
            raise ValueError("Results length doesn't match parameter values length.")

    def _get_n_states(self) -> int:
        """Get number of state variables."""
        if self.is_adaptive_study:
            return self.as_bse.sampler.state_dim
        if self.bse.y0 is None:
            return 0
        return self.bse.y0.shape[1]

    def _compute_param_bse(self, param_idx: int) -> BasinStabilityEstimator:
        """Compute BSE for a specific parameter index.

        :param param_idx: Index of the parameter value to compute.
        :return: Computed BasinStabilityEstimator instance.
        :raises ValueError: If computation fails.
        """
        try:
            param_value = self.as_bse.parameter_values[param_idx]
            param_name = list(self.as_bse.labels[0].keys())[0] if self.as_bse.labels else "param"

            logger.info(f"Computing BSE for {param_name}={param_value} (index {param_idx})")

            # Update parameter via eval (same as AS-BSE does)
            assignment = f"{param_name} = {param_value}"
            context: dict[str, object] = {
                "n": self.as_bse.n,
                "ode_system": self.as_bse.ode_system,
                "sampler": self.as_bse.sampler,
                "solver": self.as_bse.solver,
                "feature_extractor": self.as_bse.feature_extractor,
                "cluster_classifier": self.as_bse.cluster_classifier,
            }

            eval(compile(assignment, "<string>", "exec"), context, context)

            # Create and run BSE
            bse = BasinStabilityEstimator(
                n=self.as_bse.n,
                ode_system=self.as_bse.ode_system,
                sampler=self.as_bse.sampler,
                solver=self.as_bse.solver,
                feature_extractor=self.as_bse.feature_extractor,
                predictor=self.as_bse.cluster_classifier,
                feature_selector=None,
            )

            bse.estimate_bs()
            logger.info(f"BSE computation complete for {param_name}={param_value}")

            return bse

        except Exception as e:
            logger.error(f"Failed to compute BSE for parameter index {param_idx}: {e}")
            raise ValueError(f"BSE computation failed for parameter {param_value}: {e}") from e

    def _init_as_pages(self) -> None:
        """Initialize AS mode pages with AIO components."""
        self.param_manager = ASParameterManagerAIO(
            self.as_bse, self.state_labels, self._compute_param_bse
        )
        self.param_overview = ParamOverviewAIO(self.as_bse, "as-overview", self.state_labels)
        self.param_bifurcation = ParamBifurcationAIO(
            self.as_bse, "as-bifurcation", self.state_labels
        )

    def _init_bse_pages(self) -> None:
        """Initialize BSE mode pages with AIO components."""
        self.state_space = StateSpaceAIO(
            self.bse, "main-state", self.state_labels, self.options.state_space
        )
        self.feature_space = FeatureSpaceAIO(
            self.bse, "main-feature", self.state_labels, self.options.feature_space
        )
        self.basin_stability = BasinStabilityAIO(self.bse, "main-basin", self.state_labels)
        self.phase_plot = TemplatePhasePlotAIO(
            self.bse, "main-phase", False, self.state_labels, self.options.phase_plot
        )
        self.template_ts = TemplateTimeSeriesAIO(
            self.bse, "main-template", self.state_labels, self.options.template_ts
        )

    def _create_layout(self) -> dmc.MantineProvider:
        """Create the Dash app layout with navigation and page container."""
        n_states = self._get_n_states()
        initial_view = self.options.initial_view

        if self.is_adaptive_study:
            nav_items = self._build_as_nav_items(initial_view)
            initial_page_content = self.param_overview.render()
            trajectory_modal_content = html.Div(id="as-modals-container")
        else:
            nav_items = self._build_bse_nav_items(n_states, initial_view)
            page_components = {
                IDs.BASIN_STABILITY: self.basin_stability,
                IDs.STATE_SPACE: self.state_space,
                IDs.FEATURE_SPACE: self.feature_space,
                IDs.PHASE_PLOT: self.phase_plot,
                IDs.TEMPLATE_TS: self.template_ts,
            }
            component = page_components.get(initial_view, self.basin_stability)
            initial_page_content = component.render()
            trajectory_modal_content = TrajectoryModalAIO(
                self.bse, "main-modal", self.state_labels
            ).render()

        return dmc.MantineProvider(
            forceColorScheme="dark",
            children=[
                dcc.Location(id=self.URL, refresh=False),
                dmc.AppShell(
                    [
                        dmc.AppShellHeader(
                            dmc.Group(
                                [
                                    dmc.Title("pyBasin: Basin Stability Explorer", order=3),
                                ],
                                px="md",
                                h="100%",
                            ),
                        ),
                        dmc.AppShellNavbar(
                            [dmc.ScrollArea(nav_items, type="scroll")],
                            p="md",
                        ),
                        dmc.AppShellMain(
                            [
                                dmc.Box(
                                    pos="relative",
                                    style={"minHeight": "calc(100vh - 60px)"},
                                    children=[
                                        dmc.LoadingOverlay(
                                            id="page-loading-overlay",
                                            visible=False,
                                            overlayProps={"radius": "sm", "blur": 2},
                                            zIndex=1000,
                                        ),
                                        dmc.Container(
                                            [
                                                html.Div(
                                                    id=self.PAGE_CONTAINER,
                                                    children=initial_page_content,
                                                ),
                                                trajectory_modal_content,
                                            ],
                                            fluid=True,
                                            p=0,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                    header={"height": 60},
                    navbar={"width": 220, "breakpoint": "sm"},
                    padding=0,
                ),
                dcc.Store(id=self.CURRENT_VIEW, data=initial_view),
            ],
        )

    def _build_bse_nav_items(self, n_states: int, initial_view: str) -> list:
        """Build navigation items for standard BSE mode."""
        return [
            dmc.NavLink(
                label="Basin Stability",
                leftSection=html.Span("📊"),
                id="nav-bs",
                active=self._nav_active_state(initial_view == IDs.BASIN_STABILITY),
                n_clicks=0,
            ),
            dmc.NavLink(
                label="State Space",
                leftSection=html.Span("🎯"),
                id="nav-state",
                active=self._nav_active_state(initial_view == IDs.STATE_SPACE),
                n_clicks=0,
            ),
            dmc.NavLink(
                label="Feature Space",
                leftSection=html.Span("📈"),
                id="nav-feature",
                active=self._nav_active_state(initial_view == IDs.FEATURE_SPACE),
                n_clicks=0,
            ),
            dmc.Divider(label="Template Trajectories", my="sm"),
            dmc.NavLink(
                label="Phase Plot",
                leftSection=html.Span("〰️"),
                id="nav-phase",
                active=self._nav_active_state(initial_view == IDs.PHASE_PLOT),
                n_clicks=0,
            ),
            dmc.NavLink(
                label="Time Series",
                leftSection=html.Span("📉"),
                id="nav-template-ts",
                active=self._nav_active_state(initial_view == IDs.TEMPLATE_TS),
                n_clicks=0,
            ),
        ]

    def _build_as_nav_items(self, initial_view: str) -> list:
        """Build navigation items for Adaptive Study mode."""
        param_name = list(self.as_bse.labels[0].keys())[0] if self.as_bse.labels else "param"
        param_values = self.as_bse.parameter_values
        param_options = [
            {"value": str(i), "label": f"{param_name}={val:.4f}"}
            for i, val in enumerate(param_values)
        ]

        return [
            dmc.Divider(label="Parameter Study", my="sm"),
            dmc.NavLink(
                label="Parameter Variation",
                leftSection=html.Span("📊"),
                id="nav-param-overview",
                active=self._nav_active_state(initial_view == IDs.PARAM_OVERVIEW),
                n_clicks=0,
            ),
            dmc.NavLink(
                label="Bifurcation",
                leftSection=html.Span("🌀"),
                id="nav-param-bifurcation",
                active=self._nav_active_state(initial_view == IDs.PARAM_BIFURCATION),
                n_clicks=0,
            ),
            dmc.Divider(label="Parameter Value", my="sm"),
            dmc.Select(
                id="param-value-selector",
                label="Select Parameter",
                # Type is wrong. Check: https://www.dash-mantine-components.com/components/select
                data=param_options,  # type: ignore[arg-type]
                value="0",
                searchable=True,
                allowDeselect=False,
                mb="md",
            ),
            html.Div(id="param-bse-nav-items"),
        ]

    def run(self, port: int = 8050, debug: bool = False) -> None:
        """
        Launch the interactive plotter as a standalone Dash server.

        :param port: Port to run the server on (default: 8050).
        :param debug: Enable Dash debug mode (default: False).
        """
        self.app = Dash(
            __name__,
            external_stylesheets=[],
            suppress_callback_exceptions=True,
        )

        self.app.layout = self._create_layout()

        # Register navigation callbacks (using @self.app.callback for view switching)
        self._register_navigation_callbacks()

        logger.info("\n🚀 Starting Basin Stability Visualization Server")
        logger.info("   Open http://localhost:%d in your browser", port)
        logger.info("   Press Ctrl+C to stop the server\n")

        self.app.run(host="0.0.0.0", port=port, debug=debug)

    def _register_navigation_callbacks(self) -> None:
        """Register callbacks for navigation and page switching."""
        if self.app is None:
            return

        # Clientside callback to show loading overlay immediately on navigation
        self.app.clientside_callback(
            """
            function(pathname) {
                return true;
            }
            """,
            Output("page-loading-overlay", "visible", allow_duplicate=True),
            Input(self.URL, "pathname"),
            prevent_initial_call=True,
        )

        if self.is_adaptive_study:
            self._register_as_nav_callbacks()
        else:
            self._register_bse_nav_callbacks()

    def _register_bse_nav_callbacks(self) -> None:
        """Register navigation callbacks for standard BSE mode."""
        if self.app is None:
            return

        path_to_view = {
            "/": IDs.BASIN_STABILITY,
            "/basin-stability": IDs.BASIN_STABILITY,
            "/state-space": IDs.STATE_SPACE,
            "/feature-space": IDs.FEATURE_SPACE,
            "/phase": IDs.PHASE_PLOT,
            "/time-series": IDs.TEMPLATE_TS,
        }
        view_to_path = {
            IDs.BASIN_STABILITY: "/basin-stability",
            IDs.STATE_SPACE: "/state-space",
            IDs.FEATURE_SPACE: "/feature-space",
            IDs.PHASE_PLOT: "/phase",
            IDs.TEMPLATE_TS: "/time-series",
        }

        @self.app.callback(
            Output(self.URL, "pathname"),
            [
                Input("nav-bs", "n_clicks"),
                Input("nav-state", "n_clicks"),
                Input("nav-feature", "n_clicks"),
                Input("nav-phase", "n_clicks"),
                Input("nav-template-ts", "n_clicks"),
            ],
            prevent_initial_call=True,
        )
        def update_url(*_args: int) -> str:
            triggered = ctx.triggered_id
            views = {
                "nav-bs": IDs.BASIN_STABILITY,
                "nav-state": IDs.STATE_SPACE,
                "nav-feature": IDs.FEATURE_SPACE,
                "nav-phase": IDs.PHASE_PLOT,
                "nav-template-ts": IDs.TEMPLATE_TS,
            }
            view = (
                views.get(triggered, IDs.BASIN_STABILITY)
                if isinstance(triggered, str)
                else IDs.BASIN_STABILITY
            )
            return view_to_path.get(view, "/basin-stability")

        @self.app.callback(
            [
                Output(self.CURRENT_VIEW, "data"),
                Output(self.PAGE_CONTAINER, "children"),
                Output("nav-bs", "active"),
                Output("nav-state", "active"),
                Output("nav-feature", "active"),
                Output("nav-phase", "active"),
                Output("nav-template-ts", "active"),
                Output("page-loading-overlay", "visible"),
            ],
            Input(self.URL, "pathname"),
        )
        def update_page_from_url(
            pathname: str | None,
        ) -> tuple[
            str,
            html.Div,
            NavActiveState,
            NavActiveState,
            NavActiveState,
            NavActiveState,
            NavActiveState,
            bool,
        ]:
            if pathname is None:
                pathname = "/"
            view = path_to_view.get(pathname, IDs.BASIN_STABILITY)

            page_components = {
                IDs.BASIN_STABILITY: self.basin_stability,
                IDs.STATE_SPACE: self.state_space,
                IDs.FEATURE_SPACE: self.feature_space,
                IDs.PHASE_PLOT: self.phase_plot,
                IDs.TEMPLATE_TS: self.template_ts,
            }
            component = page_components.get(view, self.basin_stability)
            page_layout = component.render()

            view_to_nav = {
                IDs.BASIN_STABILITY: "nav-bs",
                IDs.STATE_SPACE: "nav-state",
                IDs.FEATURE_SPACE: "nav-feature",
                IDs.PHASE_PLOT: "nav-phase",
                IDs.TEMPLATE_TS: "nav-template-ts",
            }
            active_nav = view_to_nav.get(view, "nav-bs")

            return (
                view,
                page_layout,
                self._nav_active_state(active_nav == "nav-bs"),
                self._nav_active_state(active_nav == "nav-state"),
                self._nav_active_state(active_nav == "nav-feature"),
                self._nav_active_state(active_nav == "nav-phase"),
                self._nav_active_state(active_nav == "nav-template-ts"),
                False,
            )

    def _register_as_nav_callbacks(self) -> None:
        """Register navigation callbacks for Adaptive Study mode."""
        if self.app is None:
            return

        path_to_view = {
            "/": IDs.PARAM_OVERVIEW,
            "/param-overview": IDs.PARAM_OVERVIEW,
            "/param-bifurcation": IDs.PARAM_BIFURCATION,
        }
        view_to_path = {
            IDs.PARAM_OVERVIEW: "/param-overview",
            IDs.PARAM_BIFURCATION: "/param-bifurcation",
        }

        @self.app.callback(
            Output(self.URL, "pathname"),
            [
                Input("nav-param-overview", "n_clicks"),
                Input("nav-param-bifurcation", "n_clicks"),
            ],
            prevent_initial_call=True,
        )
        def update_url(*_args: int) -> str:
            triggered = ctx.triggered_id
            views = {
                "nav-param-overview": IDs.PARAM_OVERVIEW,
                "nav-param-bifurcation": IDs.PARAM_BIFURCATION,
            }
            view = (
                views.get(triggered, IDs.PARAM_OVERVIEW)
                if isinstance(triggered, str)
                else IDs.PARAM_OVERVIEW
            )
            return view_to_path.get(view, "/param-overview")

        @self.app.callback(
            [
                Output(self.CURRENT_VIEW, "data"),
                Output(self.PAGE_CONTAINER, "children"),
                Output("nav-param-overview", "active"),
                Output("nav-param-bifurcation", "active"),
                Output("page-loading-overlay", "visible"),
            ],
            Input(self.URL, "pathname"),
        )
        def update_page_from_url(
            pathname: str | None,
        ) -> tuple[str, html.Div, bool, bool, bool]:
            if pathname is None:
                pathname = "/"
            view = path_to_view.get(pathname, IDs.PARAM_OVERVIEW)

            page_components = {
                IDs.PARAM_OVERVIEW: self.param_overview,
                IDs.PARAM_BIFURCATION: self.param_bifurcation,
            }
            component = page_components.get(view, self.param_overview)
            page_layout = component.render()

            return (
                view,
                page_layout,
                view == IDs.PARAM_OVERVIEW,
                view == IDs.PARAM_BIFURCATION,
                False,
            )

        @self.app.callback(
            Output("param-bse-nav-items", "children"),
            Input("param-value-selector", "value"),
        )
        def update_param_bse_nav_items(param_idx_str: str | None) -> list:
            """Create BSE navigation items for selected parameter."""
            if param_idx_str is None:
                return []

            param_idx = int(param_idx_str)
            pages, _ = self.param_manager.get_or_create_pages(param_idx)

            return [
                dmc.Divider(label="Basin Stability Plots", my="sm"),
                dmc.NavLink(
                    label="Basin Stability",
                    leftSection=html.Span("📊"),
                    id={"type": "nav-param-page", "page": "basin-stability"},
                    n_clicks=0,
                ),
                dmc.NavLink(
                    label="State Space",
                    leftSection=html.Span("🎯"),
                    id={"type": "nav-param-page", "page": "state-space"},
                    n_clicks=0,
                ),
                dmc.NavLink(
                    label="Feature Space",
                    leftSection=html.Span("📈"),
                    id={"type": "nav-param-page", "page": "feature-space"},
                    n_clicks=0,
                ),
                dmc.Divider(label="Template Trajectories", my="sm"),
                dmc.NavLink(
                    label="Phase Plot",
                    leftSection=html.Span("〰️"),
                    id={"type": "nav-param-page", "page": "template-phase-plot"},
                    n_clicks=0,
                ),
                dmc.NavLink(
                    label="Time Series",
                    leftSection=html.Span("📉"),
                    id={"type": "nav-param-page", "page": "template-time-series"},
                    n_clicks=0,
                ),
            ]

        # Clientside callback to show loading when clicking param-page nav items
        self.app.clientside_callback(
            """
            function(n_clicks) {
                if (n_clicks && n_clicks.some(x => x > 0)) {
                    return true;
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output("page-loading-overlay", "visible", allow_duplicate=True),
            Input({"type": "nav-param-page", "page": ALL}, "n_clicks"),
            prevent_initial_call=True,
        )

        @self.app.callback(
            [
                Output(self.PAGE_CONTAINER, "children", allow_duplicate=True),
                Output("page-loading-overlay", "visible", allow_duplicate=True),
            ],
            [
                Input({"type": "nav-param-page", "page": ALL}, "n_clicks"),
                State("param-value-selector", "value"),
            ],
            prevent_initial_call=True,
        )
        def navigate_to_param_page(
            _n_clicks_list: list[int], param_idx_str: str | None
        ) -> tuple[Component | NoUpdate, bool | NoUpdate]:
            """Navigate to BSE page for selected parameter."""
            if param_idx_str is None or not ctx.triggered:
                return no_update, no_update

            triggered_id = ctx.triggered_id
            if isinstance(triggered_id, dict):
                page_id = triggered_id.get("page")
                if page_id is not None and isinstance(page_id, str):
                    param_idx = int(param_idx_str)
                    pages, _ = self.param_manager.get_or_create_pages(param_idx)

                    page_component = pages.get(page_id)
                    if page_component is not None:
                        return page_component.render(), False

            return no_update, no_update

    def _nav_active_state(self, is_active: bool) -> NavActiveState:
        return "exact" if is_active else None
