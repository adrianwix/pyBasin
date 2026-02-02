# pyright: basic

import logging
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.predictors.base import ClassifierPredictor
from pybasin.utils import generate_filename, resolve_folder

logger = logging.getLogger(__name__)


class MatplotlibPlotter:
    def __init__(self, bse: BasinStabilityEstimator):
        """
        Initialize the Plotter with a BasinStabilityEstimator instance.

        :param bse: An instance of BasinStabilityEstimator.
        """
        self.bse = bse

    # Do we need this if the methods save on their own alrady?
    def save_plot(self, plot_name: str):
        if self.bse.save_to is None:
            raise ValueError("save_to is not defined.")
        full_folder = resolve_folder(self.bse.save_to)
        file_name = generate_filename(plot_name, "png")
        full_path = os.path.join(full_folder, file_name)

        logger.info("Saving plots to: %s", full_path)
        plt.savefig(full_path, dpi=300)  # type: ignore[misc]

    def plot_basin_stability_bars(self, ax: Axes | None = None):
        """
        Plot basin stability values as a bar chart.

        :param ax: Matplotlib axes to plot on. If None, creates a new figure.
        """
        if self.bse.bs_vals is None:
            raise ValueError(
                "No basin stability values available. Please run estimate_bs() before plotting."
            )

        # Create standalone figure if no axes provided
        if ax is None:
            plt.figure(figsize=(6, 5))  # type: ignore[misc]
            ax = plt.gca()  # type: ignore[assignment]
            standalone = True
        else:
            standalone = False

        # Plot bar chart
        bar_labels, values = zip(*self.bse.bs_vals.items(), strict=True)
        ax.bar(bar_labels, values, color=["#ff7f0e", "#1f77b4"])  # type: ignore[misc]
        ax.set_xticks(bar_labels)  # type: ignore[misc]
        ax.set_ylabel("Fraction of samples")  # type: ignore[misc]
        ax.set_title("Basin Stability")  # type: ignore[misc]

        # Show/save if standalone
        if standalone:
            plt.tight_layout()
            # TODO: review this is always called
            if self.bse.save_to:
                self.save_plot("basin_stability_bars")
            else:
                plt.show()  # type: ignore[misc]

    def plot_state_space(self, ax: Axes | None = None):
        """
        Plot initial conditions in state space, colored by their attractor labels.

        :param ax: Matplotlib axes to plot on. If None, creates a new figure.
        """
        if self.bse.y0 is None:
            raise ValueError(
                "No initial conditions available. Please run estimate_bs() before plotting."
            )

        if self.bse.solution is None or self.bse.solution.labels is None:
            raise ValueError("No labels available. Please run estimate_bs() before plotting.")

        # Extract data
        initial_conditions = self.bse.y0.cpu().numpy()
        labels = np.array(self.bse.solution.labels)

        # Create standalone figure if no axes provided
        if ax is None:
            plt.figure(figsize=(6, 5))  # type: ignore[misc]
            ax = plt.gca()  # type: ignore[assignment]
            standalone = True
        else:
            standalone = False

        # Plot state space scatter
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = np.where(labels == label)
            ax.scatter(  # type: ignore[misc]
                initial_conditions[idx, 0],
                initial_conditions[idx, 1],
                s=4,
                alpha=0.5,
                label=str(label),
            )
        ax.set_title("Initial Conditions in State Space")  # type: ignore[misc]
        ax.set_xlabel("y_1")  # type: ignore[misc]
        ax.set_ylabel("y_2")  # type: ignore[misc]
        ax.legend(loc="upper left")  # type: ignore[misc]

        # Show/save if standalone
        if standalone:
            plt.tight_layout()
            if self.bse.save_to:
                self.save_plot("state_space")
            else:
                plt.show()  # type: ignore[misc]

    def plot_feature_space(self, ax: Axes | None = None):
        """
        Plot feature space with classifier results.

        :param ax: Matplotlib axes to plot on. If None, creates a new figure.
        """
        if self.bse.solution is None:
            raise ValueError("No solutions available. Please run estimate_bs() before plotting.")

        if self.bse.solution.features is None:
            raise ValueError("No features available. Please run estimate_bs() before plotting.")

        if self.bse.solution.labels is None:
            raise ValueError("No labels available. Please run estimate_bs() before plotting.")

        # Extract data
        features_array = self.bse.solution.features.cpu().numpy()
        all_labels = np.array(self.bse.solution.labels)

        # Features only exist for bounded trajectories.
        # Filter out unbounded trajectories (matching feature_space_aio.py approach)
        bounded_mask = all_labels != "unbounded"
        labels = all_labels[bounded_mask]

        # Verify features array matches bounded trajectory count
        if len(features_array) != len(labels):
            raise ValueError(
                f"Feature array size mismatch: {len(features_array)} features "
                f"vs {len(labels)} bounded trajectories"
            )

        n_features = features_array.shape[1] if features_array.ndim > 1 else 1

        # Ensure features_array is 2D for consistent indexing
        if features_array.ndim == 1:
            features_array = features_array.reshape(-1, 1)

        # Create standalone figure if no axes provided
        if ax is None:
            plt.figure(figsize=(6, 5))  # type: ignore[misc]
            ax = plt.gca()  # type: ignore[assignment]
            standalone = True
        else:
            standalone = False

        # Plot feature space scatter using boolean masks (matching feature_space_aio.py)
        unique_labels = np.unique(labels)
        rng = np.random.default_rng(42)

        for label in unique_labels:
            mask = labels == label
            if n_features >= 2:
                ax.scatter(  # type: ignore[misc]
                    features_array[mask, 0],
                    features_array[mask, 1],
                    s=5,
                    alpha=0.5,
                    label=str(label),
                )
            else:
                x_data = features_array[mask, 0]
                y_jitter = rng.uniform(-0.4, 0.4, size=len(x_data))
                ax.scatter(  # type: ignore[misc]
                    x_data,
                    y_jitter,
                    s=5,
                    alpha=0.5,
                    label=str(label),
                )

        if n_features >= 2:
            ax.set_title("Feature Space with Classifier Results")  # type: ignore[misc]
            ax.set_xlabel("Feature 1")  # type: ignore[misc]
            ax.set_ylabel("Feature 2")  # type: ignore[misc]
        else:
            ax.set_title("Feature Space (1D Strip Plot)")  # type: ignore[misc]
            ax.set_xlabel("Feature 1")  # type: ignore[misc]
            ax.set_ylabel("")  # type: ignore[misc]
            ax.set_yticks([])  # type: ignore[misc]
            ax.set_ylim(-0.6, 0.6)  # type: ignore[misc]

        ax.legend()  # type: ignore[misc]

        # Show/save if standalone
        if standalone:
            plt.tight_layout()
            if self.bse.save_to:
                self.save_plot("feature_space")
            else:
                plt.show()  # type: ignore[misc]

    def plot_bse_results(self):
        """
        Generate diagnostic plots using the data stored in self.solution:
            1. A bar plot of basin stability values.
            2. A scatter plot of initial conditions (state space).
            3. A scatter plot of the feature space with classifier results.
            4. A placeholder plot for future use.

        This method combines the individual plotting functions into a 2x2 grid.
        For individual plots, use plot_basin_stability_bars(), plot_state_space(),
        or plot_feature_space() directly.
        """
        # Create 2x2 subplot grid
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # type: ignore[misc]

        # Use the individual plotting functions
        self.plot_basin_stability_bars(ax=axs[0, 0])
        self.plot_state_space(ax=axs[0, 1])
        self.plot_feature_space(ax=axs[1, 0])

        # Placeholder for future plotting
        axs[1, 1].set_title("Future Plot")

        plt.tight_layout()

        # Save the combined figure
        if self.bse.save_to:
            self.save_plot("bse_results_plot")
        else:
            plt.show()  # type: ignore[misc]

    # Plots 2 states over time for the same trajectory in the same space
    def plot_templates_phase_space(self, x_var: int = 0, y_var: int = 1, z_var: int | None = None):
        """
        Plot trajectories for the template initial conditions in 2D or 3D phase space.
        """
        if not isinstance(self.bse.predictor, ClassifierPredictor):
            raise ValueError(
                "plot_phase requires a ClassifierPredictor with template initial conditions."
            )

        # Use classifier's solver if available, otherwise use main solver
        solver = self.bse.predictor.solver or self.bse.solver  # type: ignore[misc]

        # Convert template_y0 list to tensor on solver's device
        template_tensor = torch.tensor(
            self.bse.predictor.template_y0,  # type: ignore[misc]
            dtype=torch.float32,
            device=solver.device,
        )

        _t, trajectories = solver.integrate(
            self.bse.ode_system,
            template_tensor,
        )

        # Move tensors to CPU for plotting
        trajectories = trajectories.cpu()

        fig = plt.figure(figsize=(8, 6))  # type: ignore[misc]
        if z_var is None:
            ax: Axes = fig.add_subplot(111)  # type: ignore[assignment]
            for _i, (label, traj) in enumerate(
                zip(self.bse.predictor.labels, trajectories.permute(1, 0, 2), strict=True)  # type: ignore[arg-type]
            ):
                ax.plot(traj[:, x_var], traj[:, y_var], label=str(label))  # type: ignore[misc]
            ax.set_xlabel(f"State {x_var}")  # type: ignore[misc]
            ax.set_ylabel(f"State {y_var}")  # type: ignore[misc]
            ax.set_title("2D Phase Plot")  # type: ignore[misc]
        else:
            ax = fig.add_subplot(111, projection="3d")  # type: ignore[assignment]
            for _i, (label, traj) in enumerate(
                zip(self.bse.predictor.labels, trajectories.permute(1, 0, 2), strict=True)  # type: ignore[arg-type]
            ):
                ax.plot(traj[:, x_var], traj[:, y_var], traj[:, z_var], label=str(label))  # type: ignore[misc,attr-defined]
            ax.set_xlabel(f"State {x_var}")  # type: ignore[misc]
            ax.set_ylabel(f"State {y_var}")  # type: ignore[misc]
            ax.set_zlabel(f"State {z_var}")  # type: ignore[misc,attr-defined]
            ax.set_title("3D Phase Plot")  # type: ignore[misc]

        plt.legend()  # type: ignore[misc]
        plt.tight_layout()

        if self.bse.save_to:
            self.save_plot("phase_plot")
        else:
            plt.show()  # type: ignore[misc]

    # Plots over time
    def plot_templates_trajectories(
        self, plotted_var: int, time_span: tuple[float, float] | None = None
    ):
        """
        Plot trajectories for the template initial conditions.

        :param plotted_var: Index of the variable to plot.
        :param time_span: Time range to plot (t_start, t_end).
        """
        if not isinstance(self.bse.predictor, ClassifierPredictor):
            raise ValueError(
                "plot_templates requires a ClassifierPredictor with template initial conditions."
            )

        # Use classifier's solver if available, otherwise use main solver
        solver = self.bse.predictor.solver or self.bse.solver  # type: ignore[misc]

        # Convert template_y0 list to tensor on solver's device
        template_tensor = torch.tensor(
            self.bse.predictor.template_y0,  # type: ignore[misc]
            dtype=torch.float32,
            device=solver.device,
        )

        # Get trajectories for template initial conditions
        t, y = solver.integrate(
            self.bse.ode_system,
            template_tensor,
        )

        # Move tensors to CPU for plotting
        t = t.cpu()
        y = y.cpu()

        plt.figure(figsize=(8, 6))  # type: ignore[misc]

        # Filter time if specified
        idx = (t >= time_span[0]) & (t <= time_span[1]) if time_span is not None else slice(None)

        # Plot each trajectory
        # Use permute instead of transpose for 3D tensors
        for _i, (label, traj) in enumerate(
            zip(self.bse.predictor.labels, y.permute(1, 0, 2), strict=True)  # type: ignore[arg-type]
        ):
            plt.plot(t[idx], traj[idx, plotted_var], label=f"{label}")  # type: ignore[misc]

        plt.xlabel("Time")  # type: ignore[misc]
        plt.ylabel(f"State {plotted_var}")  # type: ignore[misc]
        plt.title("Template Trajectories")  # type: ignore[misc]
        plt.legend()  # type: ignore[misc]
        plt.grid(True)  # type: ignore[misc]

        # Save plot
        if self.bse.save_to:
            self.save_plot("template_trajectories_plot")
        else:
            plt.show()  # type: ignore[misc]
