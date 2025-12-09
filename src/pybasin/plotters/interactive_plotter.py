# pyright: basic
"""Interactive web-based plotter using Dash and Plotly with Mantine components."""

import logging
from typing import Literal

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
from dash import Dash, Input, Output, ctx, dcc, html
from dash.exceptions import PreventUpdate

from pybasin.as_basin_stability_estimator import ASBasinStabilityEstimator
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.base_page import BasePage
from pybasin.plotters.basin_stability_page import BasinStabilityPage
from pybasin.plotters.feature_space_page import FeatureSpacePage
from pybasin.plotters.feature_space_page import set_page_instance as set_feature_instance
from pybasin.plotters.ids import IDs
from pybasin.plotters.phase_plot_page import PhasePlotPage
from pybasin.plotters.phase_plot_page import set_page_instance as set_phase_instance
from pybasin.plotters.state_space_page import StateSpacePage
from pybasin.plotters.state_space_page import set_page_instance as set_state_instance
from pybasin.plotters.template_time_series_page import TemplateTimeSeriesPage
from pybasin.plotters.template_time_series_page import (
    set_page_instance as set_template_ts_instance,
)
from pybasin.plotters.trajectory_modal import (
    SELECTED_SAMPLE_DATA,
    TrajectoryModal,
    set_modal_instance,
)
from pybasin.plotters.types import InteractivePlotterOptions

logger = logging.getLogger(__name__)


NavActiveState = Literal["exact", "partial"] | None


class InteractivePlotter:
    """
    Interactive web-based plotter for basin stability visualization.

    Uses Dash with Mantine components for a modern UI and Plotly for
    interactive visualizations. Each page owns its controls, plot, and callbacks.

    Attributes:
        bse: BasinStabilityEstimator instance with computed results.
        state_labels: Optional mapping of state indices to custom labels.
        app: Dash application instance.
    """

    # Main layout IDs
    PAGE_CONTAINER = "page-container"
    CURRENT_VIEW = "current-view"
    URL = "url"

    def __init__(
        self,
        bse: BasinStabilityEstimator | ASBasinStabilityEstimator,
        state_labels: dict[int, str] | None = None,
        options: InteractivePlotterOptions | None = None,
    ):
        """
        Initialize the InteractivePlotter.

        :param bse: BasinStabilityEstimator or ASBasinStabilityEstimator instance.
        :param state_labels: Optional dict mapping state indices to labels,
                            e.g., {0: "θ", 1: "ω"} for a pendulum system.
        :param options: Optional configuration for default control values.
        """
        self.is_adaptive_study = isinstance(bse, ASBasinStabilityEstimator)

        if self.is_adaptive_study:
            self.as_bse: ASBasinStabilityEstimator = bse  # type: ignore[assignment]
            self.current_param_bse: BasinStabilityEstimator | None = None
        else:
            self.bse: BasinStabilityEstimator = bse  # type: ignore[assignment]

        self.state_labels = state_labels or {}
        self.options = options or InteractivePlotterOptions()

        # Override initial view for AS mode if not explicitly set
        if self.is_adaptive_study and options is None:
            self.options.initial_view = IDs.PARAM_OVERVIEW

        self.app: Dash | None = None

        if self.is_adaptive_study:
            self._validate_as_bse()
        else:
            self._validate_bse()

        self._init_pages()

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
            param_name = self.as_bse.as_params["adaptative_parameter_name"]

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
                cluster_classifier=self.as_bse.cluster_classifier,
                feature_selector=None,
            )

            bse.estimate_bs()
            logger.info(f"BSE computation complete for {param_name}={param_value}")

            return bse

        except Exception as e:
            logger.error(f"Failed to compute BSE for parameter index {param_idx}: {e}")
            raise ValueError(f"BSE computation failed for parameter {param_value}: {e}") from e

    def _init_pages(self) -> None:
        """Initialize all page modules and register them for callbacks."""
        if self.is_adaptive_study:
            # Import AS page modules
            from pybasin.plotters.parameter_bifurcation_page import (
                ParamBifurcationPage,
                set_page_instance as set_bifurcation_instance,
            )
            from pybasin.plotters.parameter_overview_page import (
                ParamOverviewPage,
                set_page_instance as set_overview_instance,
            )

            # Create AS-specific pages
            self.param_overview_page = ParamOverviewPage(self.as_bse, self.state_labels)
            self.param_bifurcation_page = ParamBifurcationPage(self.as_bse, self.state_labels)

            # Register AS page instances
            set_overview_instance(self.param_overview_page)
            set_bifurcation_instance(self.param_bifurcation_page)

            # Create placeholder trajectory modal (will be built later when BSE is computed)
            self.trajectory_modal_placeholder = html.Div(id="trajectory-modal-placeholder")

            # Map AS page IDs (BSE pages added dynamically after param computation)
            self._pages = {
                IDs.PARAM_OVERVIEW: self.param_overview_page,
                IDs.PARAM_BIFURCATION: self.param_bifurcation_page,
            }
        else:
            # Standard BSE mode initialization
            self.bs_page = BasinStabilityPage(self.bse, self.state_labels)
            self.state_page = StateSpacePage(
                self.bse, self.state_labels, options=self.options.state_space
            )
            self.feature_page = FeatureSpacePage(
                self.bse, self.state_labels, options=self.options.feature_space
            )
            self.phase_2d_page = PhasePlotPage(
                self.bse, self.state_labels, is_3d=False, options=self.options.phase_plot
            )
            self.phase_3d_page = PhasePlotPage(
                self.bse, self.state_labels, is_3d=True, options=self.options.phase_plot
            )
            self.template_ts_page = TemplateTimeSeriesPage(
                self.bse, self.state_labels, options=self.options.template_ts
            )
            self.trajectory_modal = TrajectoryModal(self.bse, self.state_labels)

            # Register page instances for module-level callbacks
            set_state_instance(self.state_page)
            set_feature_instance(self.feature_page)
            set_phase_instance(self.phase_2d_page)
            set_phase_instance(self.phase_3d_page)
            set_template_ts_instance(self.template_ts_page)
            set_modal_instance(self.trajectory_modal)

            # Map page_id to page instance
            self._pages: dict[str, BasePage] = {
                IDs.BASIN_STABILITY: self.bs_page,
                IDs.STATE_SPACE: self.state_page,
                IDs.FEATURE_SPACE: self.feature_page,
                IDs.PHASE_2D: self.phase_2d_page,
                IDs.PHASE_3D: self.phase_3d_page,
                IDs.TEMPLATE_TS: self.template_ts_page,
            }

    def _create_layout(self) -> dmc.MantineProvider:
        """Create the Dash app layout with navigation and page container."""
        n_states = self._get_n_states()
        initial_view = self.options.initial_view

        # Build navigation items conditionally based on mode
        if self.is_adaptive_study:
            nav_items = self._build_as_nav_items(initial_view)
            # For AS mode, show loading indicator initially
            initial_page_content = html.Div(
                [
                    dmc.Center(
                        [
                            dmc.Stack(
                                [
                                    dmc.Loader(size="xl", type="dots"),
                                    dmc.Text("Loading data...", size="lg", mt="md"),
                                ],
                                align="center",
                            )
                        ],
                        style={"minHeight": "60vh"},
                    )
                ]
            )
        else:
            nav_items = self._build_bse_nav_items(n_states, initial_view)
            initial_page_content = self._pages[initial_view].build_layout()  # type: ignore[index]

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
                                dmc.Container(
                                    [
                                        # Loading overlay for AS mode
                                        html.Div(
                                            [
                                                dmc.LoadingOverlay(
                                                    visible=False,
                                                    id="loading-overlay",
                                                    overlayProps={"radius": "sm", "blur": 2},
                                                    loaderProps={"type": "dots", "size": "xl"},
                                                    zIndex=1000,
                                                ),
                                                html.Div(
                                                    id=self.PAGE_CONTAINER,
                                                    children=initial_page_content,
                                                ),
                                                dmc.Text(
                                                    id="loading-message",
                                                    ta="center",
                                                    c="gray",
                                                    size="sm",
                                                    mt="md",
                                                ),
                                            ],
                                            style={"position": "relative", "minHeight": "100vh"},
                                        ),
                                        # Shared trajectory modal
                                        (
                                            self.trajectory_modal.build_layout()
                                            if not self.is_adaptive_study
                                            else self.trajectory_modal_placeholder
                                        ),
                                    ],
                                    fluid=True,
                                    p="md",
                                ),
                            ],
                        ),
                    ],
                    header={"height": 60},
                    navbar={"width": 220, "breakpoint": "sm"},
                    padding="md",
                ),
                # Global stores
                dcc.Store(id=self.CURRENT_VIEW, data=initial_view),
                dcc.Store(id=SELECTED_SAMPLE_DATA, data=None),
            ]
            + (
                [
                    dcc.Store(id="current-param-idx", data=0),
                    dcc.Store(id="param-bse-data", data=None),
                    dcc.Store(id="param-computation-trigger", data=0),
                    dcc.Store(id="running-computation", data=None),
                ]
                if self.is_adaptive_study
                else []
            ),
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
                label="Phase Plot 2D",
                leftSection=html.Span("〰️"),
                id="nav-phase-2d",
                active=self._nav_active_state(initial_view == IDs.PHASE_2D),
                n_clicks=0,
            ),
            dmc.NavLink(
                label="Phase Plot 3D",
                leftSection=html.Span("🌀"),
                id="nav-phase-3d",
                active=self._nav_active_state(initial_view == IDs.PHASE_3D),
                n_clicks=0,
                disabled=n_states < 3,
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
        param_name = self.as_bse.as_params["adaptative_parameter_name"].split(".")[-1]
        param_options = [
            {"label": f"{val:.4g}", "value": str(i)}
            for i, val in enumerate(self.as_bse.parameter_values)
        ]

        return [
            dmc.Select(
                id="param-selector",
                label=param_name,
                data=param_options,  # pyright: ignore[reportArgumentType]
                value="0",
                styles={"label": {"marginBottom": "8px"}},
            ),
            dmc.Divider(label="Parameter Study", my="sm"),
            dmc.NavLink(
                label="Overview",
                leftSection=html.Span("📊"),
                id="nav-param-overview",
                active=None,
                n_clicks=0,
            ),
            dmc.NavLink(
                label="Bifurcation",
                leftSection=html.Span("🌀"),
                id="nav-param-bifurcation",
                active=None,
                n_clicks=0,
            ),
            dmc.Divider(id="param-value-divider", label="Loading...", my="sm"),
            dmc.NavLink(
                label="Basin Stability",
                leftSection=html.Span("📊"),
                id="nav-bs",
                active=None,
                n_clicks=0,
                disabled=True,
            ),
            dmc.NavLink(
                label="State Space",
                leftSection=html.Span("🎯"),
                id="nav-state",
                active=None,
                n_clicks=0,
                disabled=True,
            ),
            dmc.NavLink(
                label="Feature Space",
                leftSection=html.Span("📈"),
                id="nav-feature",
                active=None,
                n_clicks=0,
                disabled=True,
            ),
            dmc.Divider(label="Template Trajectories", my="sm"),
            dmc.NavLink(
                label="Phase Plot 2D",
                leftSection=html.Span("〰️"),
                id="nav-phase-2d",
                active=None,
                n_clicks=0,
                disabled=True,
            ),
            dmc.NavLink(
                label="Phase Plot 3D",
                leftSection=html.Span("🌀"),
                id="nav-phase-3d",
                active=None,
                n_clicks=0,
                disabled=True,
            ),
            dmc.NavLink(
                label="Time Series",
                leftSection=html.Span("📉"),
                id="nav-template-ts",
                active=None,
                n_clicks=0,
                disabled=True,
            ),
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

        print("\n🚀 Starting Basin Stability Visualization Server")
        print(f"   Open http://localhost:{port} in your browser")
        print("   Press Ctrl+C to stop the server\n")

        self.app.run(host="0.0.0.0", port=port, debug=debug)

    def _register_navigation_callbacks(self) -> None:
        """Register callbacks for navigation and page switching."""
        if self.app is None:
            return

        if self.is_adaptive_study:
            self._register_as_callbacks()
        else:
            self._register_bse_callbacks()

    def _register_bse_callbacks(self) -> None:
        """Register navigation callbacks for standard BSE mode."""
        if self.app is None:
            return

        # Map between URL paths and view IDs
        path_to_view = {
            "/": IDs.BASIN_STABILITY,
            "/basin-stability": IDs.BASIN_STABILITY,
            "/state-space": IDs.STATE_SPACE,
            "/feature-space": IDs.FEATURE_SPACE,
            "/phase-2d": IDs.PHASE_2D,
            "/phase-3d": IDs.PHASE_3D,
            "/time-series": IDs.TEMPLATE_TS,
        }
        view_to_path = {
            IDs.BASIN_STABILITY: "/basin-stability",
            IDs.STATE_SPACE: "/state-space",
            IDs.FEATURE_SPACE: "/feature-space",
            IDs.PHASE_2D: "/phase-2d",
            IDs.PHASE_3D: "/phase-3d",
            IDs.TEMPLATE_TS: "/time-series",
        }

        @self.app.callback(
            Output(self.URL, "pathname"),
            [
                Input("nav-bs", "n_clicks"),
                Input("nav-state", "n_clicks"),
                Input("nav-feature", "n_clicks"),
                Input("nav-phase-2d", "n_clicks"),
                Input("nav-phase-3d", "n_clicks"),
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
                "nav-phase-2d": IDs.PHASE_2D,
                "nav-phase-3d": IDs.PHASE_3D,
                "nav-template-ts": IDs.TEMPLATE_TS,
            }
            view = views.get(triggered, IDs.BASIN_STABILITY)  # type: ignore[arg-type]
            return view_to_path.get(view, "/basin-stability")

        @self.app.callback(
            [
                Output(self.CURRENT_VIEW, "data"),
                Output(self.PAGE_CONTAINER, "children"),
                Output("nav-bs", "active"),
                Output("nav-state", "active"),
                Output("nav-feature", "active"),
                Output("nav-phase-2d", "active"),
                Output("nav-phase-3d", "active"),
                Output("nav-template-ts", "active"),
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
            NavActiveState,
        ]:
            if pathname is None:
                pathname = "/"
            view = path_to_view.get(pathname, IDs.BASIN_STABILITY)

            # Get page layout
            page: BasePage = self._pages.get(view, self.bs_page)
            page_layout: html.Div = page.build_layout()

            # Determine active nav states
            view_to_nav = {
                IDs.BASIN_STABILITY: "nav-bs",
                IDs.STATE_SPACE: "nav-state",
                IDs.FEATURE_SPACE: "nav-feature",
                IDs.PHASE_2D: "nav-phase-2d",
                IDs.PHASE_3D: "nav-phase-3d",
                IDs.TEMPLATE_TS: "nav-template-ts",
            }
            active_nav = view_to_nav.get(view, "nav-bs")

            nav_bs_active = self._nav_active_state(active_nav == "nav-bs")
            nav_state_active = self._nav_active_state(active_nav == "nav-state")
            nav_feature_active = self._nav_active_state(active_nav == "nav-feature")
            nav_phase_2d_active = self._nav_active_state(active_nav == "nav-phase-2d")
            nav_phase_3d_active = self._nav_active_state(active_nav == "nav-phase-3d")
            nav_template_ts_active = self._nav_active_state(active_nav == "nav-template-ts")

            return (
                view,
                page_layout,
                nav_bs_active,
                nav_state_active,
                nav_feature_active,
                nav_phase_2d_active,
                nav_phase_3d_active,
                nav_template_ts_active,
            )

    def _register_as_callbacks(self) -> None:
        """Register navigation callbacks for Adaptive Study mode."""
        if self.app is None:
            return

        from urllib.parse import parse_qs

        # Map between URL paths and view IDs
        path_to_view = {
            "/": IDs.PARAM_OVERVIEW,
            "/param-overview": IDs.PARAM_OVERVIEW,
            "/param-bifurcation": IDs.PARAM_BIFURCATION,
            "/basin-stability": IDs.BASIN_STABILITY,
            "/state-space": IDs.STATE_SPACE,
            "/feature-space": IDs.FEATURE_SPACE,
            "/phase-2d": IDs.PHASE_2D,
            "/phase-3d": IDs.PHASE_3D,
            "/time-series": IDs.TEMPLATE_TS,
        }
        view_to_path = {
            IDs.PARAM_OVERVIEW: "/param-overview",
            IDs.PARAM_BIFURCATION: "/param-bifurcation",
            IDs.BASIN_STABILITY: "/basin-stability",
            IDs.STATE_SPACE: "/state-space",
            IDs.FEATURE_SPACE: "/feature-space",
            IDs.PHASE_2D: "/phase-2d",
            IDs.PHASE_3D: "/phase-3d",
            IDs.TEMPLATE_TS: "/time-series",
        }

        # Callback 1: Parse URL query parameter on load
        @self.app.callback(
            Output("param-selector", "value", allow_duplicate=True),
            Input(self.URL, "search"),
            prevent_initial_call=True,
        )
        def parse_url_param(search: str | None) -> str:
            # Only update from URL if triggered by URL change (e.g., page load, manual URL edit)
            if search and "param=" in search:
                try:
                    parsed = parse_qs(search.lstrip("?"))
                    param_idx = int(parsed.get("param", ["0"])[0])
                    clamped = max(0, min(param_idx, len(self.as_bse.parameter_values) - 1))
                    return str(clamped)
                except (ValueError, IndexError):
                    pass
            return "0"

        # Callback 2: Update current param index when parameter selector changes
        @self.app.callback(
            [
                Output("current-param-idx", "data"),
                Output(self.URL, "search"),
            ],
            Input("param-selector", "value"),
            prevent_initial_call=True,
        )
        def update_param_from_selector(value: str) -> tuple[int, str]:
            param_idx = int(value)
            return param_idx, f"?param={param_idx}"

        # Callback 3: Trigger BSE computation and track it
        @self.app.callback(
            [
                Output("param-computation-trigger", "data"),
                Output("running-computation", "data"),
            ],
            Input("current-param-idx", "data"),
            prevent_initial_call=False,
        )
        def trigger_computation(param_idx: int) -> tuple[int, int]:
            return param_idx, param_idx

        # Callback 3.5: Show loading overlay when computation starts
        @self.app.callback(
            Output("loading-overlay", "visible", allow_duplicate=True),
            Input("param-computation-trigger", "data"),
            prevent_initial_call=True,
        )
        def show_loading(_trigger: int) -> bool:
            return True

        # Callback 4: Compute BSE in background (simplified - synchronous for now)
        @self.app.callback(
            [
                Output("param-bse-data", "data"),
                Output("loading-overlay", "visible"),
                Output("loading-message", "children"),
            ],
            Input("param-computation-trigger", "data"),
            prevent_initial_call=False,
        )
        def compute_param_bse(param_idx: int) -> tuple[dict, bool, str]:
            if param_idx is None:
                param_idx = 0

            try:
                # Compute BSE
                bse = self._compute_param_bse(param_idx)
                self.current_param_bse = bse

                # Return success (loading will be hidden by next callback)
                return (
                    {"success": True, "param_idx": param_idx},
                    False,
                    "",
                )
            except Exception as e:
                logger.error(f"BSE computation failed: {e}")
                return (
                    {
                        "success": False,
                        "param_idx": param_idx,
                        "error": str(e),
                    },
                    False,
                    "",
                )

        # Callback 5: Update page content and navigation after BSE computation
        @self.app.callback(
            [
                Output(self.PAGE_CONTAINER, "children", allow_duplicate=True),
                Output("param-value-divider", "label"),
                Output("nav-bs", "disabled"),
                Output("nav-state", "disabled"),
                Output("nav-feature", "disabled"),
                Output("nav-phase-2d", "disabled"),
                Output("nav-phase-3d", "disabled"),
                Output("nav-template-ts", "disabled"),
                Output("nav-param-overview", "active"),
                Output("nav-param-bifurcation", "active"),
                Output("nav-bs", "active"),
                Output("nav-state", "active"),
                Output("nav-feature", "active"),
                Output("nav-phase-2d", "active"),
                Output("nav-phase-3d", "active"),
                Output("nav-template-ts", "active"),
                Output("trajectory-modal-placeholder", "children"),
            ],
            [
                Input("param-bse-data", "data"),
                Input(self.URL, "pathname"),
                Input("running-computation", "data"),
            ],
            prevent_initial_call="initial_duplicate",
        )
        def update_page_after_computation(
            data: dict | None, pathname: str | None, running_computation: int | None
        ) -> tuple[
            html.Div,
            str,
            bool,
            bool,
            bool,
            bool,
            bool,
            bool,
            bool,
            bool,
            bool,
            bool,
            bool,
            bool,
            bool,
            bool,
            html.Div,
        ]:
            print("[DEBUG] update_page_after_computation called")
            print(f"[DEBUG]   data={data}")
            print(f"[DEBUG]   pathname={pathname}")
            print(f"[DEBUG]   running_computation={running_computation}")
            print(f"[DEBUG]   ctx.triggered_id={ctx.triggered_id}")

            # If computation data exists but doesn't match the currently running computation, ignore it
            if (
                data
                and data.get("success")
                and running_computation is not None
                and data.get("param_idx") != running_computation
            ):
                print(
                    f"[DEBUG] Ignoring stale computation for param_idx={data.get('param_idx')}, current is {running_computation}"
                )
                raise PreventUpdate

            if data is None or not data.get("success", False):
                # Show error or initial state
                error_msg = data.get("error", "Unknown error") if data else "Initializing..."
                param_idx = data.get("param_idx", 0) if data else 0
                param_value = self.as_bse.parameter_values[param_idx]

                # Get current view from pathname
                if pathname is None:
                    pathname = "/"
                view = path_to_view.get(pathname, IDs.PARAM_OVERVIEW)

                print(f"[DEBUG] Showing initial/error state for param {param_value}, view={view}")

                if data and not data.get("success", False):
                    content = html.Div(
                        [
                            dmc.Alert(
                                title="Computation Error",
                                color="red",
                                children=f"Failed to compute basin stability for parameter "
                                f"{param_value:.4g}: {error_msg}",
                            )
                        ]
                    )
                else:
                    # Show loading indicator
                    content = html.Div(
                        [
                            dmc.Center(
                                [
                                    dmc.Stack(
                                        [
                                            dmc.Loader(size="xl", type="dots"),
                                            dmc.Text(
                                                "Computing basin stability...", size="lg", mt="md"
                                            ),
                                            dmc.Text(
                                                f"Parameter = {param_value:.4g}",
                                                size="sm",
                                                c="gray",
                                            ),
                                        ],
                                        align="center",
                                    )
                                ],
                                style={"minHeight": "60vh"},
                            )
                        ]
                    )

                return (
                    content,
                    f"Parameter = {param_value:.4g}",
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    view == IDs.PARAM_OVERVIEW,
                    view == IDs.PARAM_BIFURCATION,
                    view == IDs.BASIN_STABILITY,
                    view == IDs.STATE_SPACE,
                    view == IDs.FEATURE_SPACE,
                    view == IDs.PHASE_2D,
                    view == IDs.PHASE_3D,
                    view == IDs.TEMPLATE_TS,
                    html.Div(),
                )

            # BSE computation successful - update pages
            param_idx = data["param_idx"]
            param_value = self.as_bse.parameter_values[param_idx]

            # Always recreate BSE-specific pages with new data
            from pybasin.plotters.basin_stability_page import BasinStabilityPage
            from pybasin.plotters.feature_space_page import FeatureSpacePage
            from pybasin.plotters.feature_space_page import (
                set_page_instance as set_feature_instance,
            )
            from pybasin.plotters.state_space_page import StateSpacePage
            from pybasin.plotters.state_space_page import (
                set_page_instance as set_state_instance,
            )

            print(f"[DEBUG] Creating pages with BSE: {self.current_param_bse}")
            if self.current_param_bse is not None:
                print(f"[DEBUG] BSE ode_system params: {self.current_param_bse.ode_system.params}")

            self.bs_page = BasinStabilityPage(
                self.current_param_bse,
                self.state_labels,  # type: ignore[arg-type]
                id_suffix=str(param_idx),
            )
            self.state_page = StateSpacePage(
                self.current_param_bse,  # type: ignore[arg-type]
                self.state_labels,
                options=self.options.state_space,
                id_suffix=str(param_idx),
            )
            self.feature_page = FeatureSpacePage(
                self.current_param_bse,  # type: ignore[arg-type]
                self.state_labels,
                options=self.options.feature_space,
                id_suffix=str(param_idx),
            )
            self.phase_2d_page = PhasePlotPage(
                self.current_param_bse,  # type: ignore[arg-type]
                self.state_labels,
                is_3d=False,
                options=self.options.phase_plot,
            )
            self.phase_3d_page = PhasePlotPage(
                self.current_param_bse,  # type: ignore[arg-type]
                self.state_labels,
                is_3d=True,
                options=self.options.phase_plot,
            )
            self.template_ts_page = TemplateTimeSeriesPage(
                self.current_param_bse,  # type: ignore[arg-type]
                self.state_labels,
                options=self.options.template_ts,
            )

            set_state_instance(self.state_page)
            set_feature_instance(self.feature_page)
            set_phase_instance(self.phase_2d_page)
            set_phase_instance(self.phase_3d_page)
            set_template_ts_instance(self.template_ts_page)

            self._pages[IDs.BASIN_STABILITY] = self.bs_page  # type: ignore[assignment]
            self._pages[IDs.STATE_SPACE] = self.state_page  # type: ignore[assignment]
            self._pages[IDs.FEATURE_SPACE] = self.feature_page  # type: ignore[assignment]
            self._pages[IDs.PHASE_2D] = self.phase_2d_page  # type: ignore[assignment]
            self._pages[IDs.PHASE_3D] = self.phase_3d_page  # type: ignore[assignment]
            self._pages[IDs.TEMPLATE_TS] = self.template_ts_page  # type: ignore[assignment]

            # Update trajectory modal
            from pybasin.plotters.trajectory_modal import set_modal_instance

            self.trajectory_modal = TrajectoryModal(
                self.current_param_bse,  # type: ignore[arg-type]
                self.state_labels,  # type: ignore[arg-type]
            )
            set_modal_instance(self.trajectory_modal)

            # Get current view and render page
            if pathname is None:
                pathname = "/"
            view = path_to_view.get(pathname, IDs.PARAM_OVERVIEW)

            page = self._pages.get(view, self.param_overview_page)  # type: ignore[assignment]
            page_layout = page.build_layout()  # type: ignore[union-attr]

            # Check if 3D is available
            n_states = self._get_n_states()
            phase_3d_disabled = n_states < 3

            # Wrap in div with unique ID to force React re-render
            unique_layout = html.Div(
                [
                    html.Div(id=f"timestamp-{param_idx}", style={"display": "none"}),
                    page_layout,
                ],
                id=f"page-wrapper-{param_idx}",
            )

            return (
                unique_layout,
                f"Parameter = {param_value:.4g}",
                False,
                False,
                False,
                False,
                phase_3d_disabled,
                False,
                view == IDs.PARAM_OVERVIEW,
                view == IDs.PARAM_BIFURCATION,
                view == IDs.BASIN_STABILITY,
                view == IDs.STATE_SPACE,
                view == IDs.FEATURE_SPACE,
                view == IDs.PHASE_2D,
                view == IDs.PHASE_3D,
                view == IDs.TEMPLATE_TS,
                html.Div([self.trajectory_modal.build_layout()]),
            )

        # Callback 6: Handle navigation clicks
        @self.app.callback(
            Output(self.URL, "pathname"),
            [
                Input("nav-param-overview", "n_clicks"),
                Input("nav-param-bifurcation", "n_clicks"),
                Input("nav-bs", "n_clicks"),
                Input("nav-state", "n_clicks"),
                Input("nav-feature", "n_clicks"),
                Input("nav-phase-2d", "n_clicks"),
                Input("nav-phase-3d", "n_clicks"),
                Input("nav-template-ts", "n_clicks"),
            ],
            prevent_initial_call=True,
        )
        def update_url(*_args: int) -> str:
            triggered = ctx.triggered_id
            views = {
                "nav-param-overview": IDs.PARAM_OVERVIEW,
                "nav-param-bifurcation": IDs.PARAM_BIFURCATION,
                "nav-bs": IDs.BASIN_STABILITY,
                "nav-state": IDs.STATE_SPACE,
                "nav-feature": IDs.FEATURE_SPACE,
                "nav-phase-2d": IDs.PHASE_2D,
                "nav-phase-3d": IDs.PHASE_3D,
                "nav-template-ts": IDs.TEMPLATE_TS,
            }
            view = views.get(triggered, IDs.PARAM_OVERVIEW)  # type: ignore[arg-type]
            return view_to_path.get(view, "/param-overview")

    def _nav_active_state(self, is_active: bool) -> NavActiveState:
        return "exact" if is_active else None
