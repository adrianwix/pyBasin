# pyright: basic
"""Interactive web-based plotter using Dash and Plotly with Mantine components."""

import logging
from typing import Literal

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
from dash import Dash, Input, Output, ctx, dcc, html

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
        bse: BasinStabilityEstimator,
        state_labels: dict[int, str] | None = None,
        options: InteractivePlotterOptions | None = None,
    ):
        """
        Initialize the InteractivePlotter.

        :param bse: BasinStabilityEstimator instance with computed results.
        :param state_labels: Optional dict mapping state indices to labels,
                            e.g., {0: "θ", 1: "ω"} for a pendulum system.
        :param options: Optional configuration for default control values.
        """
        self.bse = bse
        self.state_labels = state_labels or {}
        self.options = options or InteractivePlotterOptions()
        self.app: Dash | None = None
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

    def _get_n_states(self) -> int:
        """Get number of state variables."""
        if self.bse.y0 is None:
            return 0
        return self.bse.y0.shape[1]

    def _init_pages(self) -> None:
        """Initialize all page modules and register them for callbacks."""
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
                            [
                                dmc.ScrollArea(
                                    [
                                        dmc.NavLink(
                                            label="Basin Stability",
                                            leftSection=html.Span("📊"),
                                            id="nav-bs",
                                            active=self._nav_active_state(
                                                initial_view == IDs.BASIN_STABILITY
                                            ),
                                            n_clicks=0,
                                        ),
                                        dmc.NavLink(
                                            label="State Space",
                                            leftSection=html.Span("🎯"),
                                            id="nav-state",
                                            active=self._nav_active_state(
                                                initial_view == IDs.STATE_SPACE
                                            ),
                                            n_clicks=0,
                                        ),
                                        dmc.NavLink(
                                            label="Feature Space",
                                            leftSection=html.Span("📈"),
                                            id="nav-feature",
                                            active=self._nav_active_state(
                                                initial_view == IDs.FEATURE_SPACE
                                            ),
                                            n_clicks=0,
                                        ),
                                        dmc.Divider(label="Template Trajectories", my="sm"),
                                        dmc.NavLink(
                                            label="Phase Plot 2D",
                                            leftSection=html.Span("〰️"),
                                            id="nav-phase-2d",
                                            active=self._nav_active_state(
                                                initial_view == IDs.PHASE_2D
                                            ),
                                            n_clicks=0,
                                        ),
                                        dmc.NavLink(
                                            label="Phase Plot 3D",
                                            leftSection=html.Span("🌀"),
                                            id="nav-phase-3d",
                                            active=self._nav_active_state(
                                                initial_view == IDs.PHASE_3D
                                            ),
                                            n_clicks=0,
                                            disabled=n_states < 3,
                                        ),
                                        dmc.NavLink(
                                            label="Time Series",
                                            leftSection=html.Span("📉"),
                                            id="nav-template-ts",
                                            active=self._nav_active_state(
                                                initial_view == IDs.TEMPLATE_TS
                                            ),
                                            n_clicks=0,
                                        ),
                                    ],
                                    type="scroll",
                                )
                            ],
                            p="md",
                        ),
                        dmc.AppShellMain(
                            [
                                dmc.Container(
                                    [
                                        # Dynamic page container - populated by callback
                                        html.Div(
                                            id=self.PAGE_CONTAINER,
                                            children=self._pages[initial_view].build_layout(),
                                        ),
                                        # Shared trajectory modal
                                        self.trajectory_modal.build_layout(),
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
            ],
        )

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

    def _nav_active_state(self, is_active: bool) -> NavActiveState:
        return "exact" if is_active else None
