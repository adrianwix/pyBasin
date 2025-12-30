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

    def save_plot(self, plot_name: str):
        if self.bse.save_to is None:
            raise ValueError("save_to is not defined.")
        full_folder = resolve_folder(self.bse.save_to)
        file_name = generate_filename(plot_name, "png")
        full_path = os.path.join(full_folder, file_name)

        logger.info("Saving plots to: %s", full_path)
        plt.savefig(full_path, dpi=300)  # type: ignore[misc]

    def plot_bse_results(self):
        """
        Generate diagnostic plots using the data stored in self.solution:
            1. A bar plot of basin stability values.
            2. A scatter plot of initial conditions (state space).
            3. A scatter plot of the feature space with classifier results.
            4. A placeholder plot for future use.
        """
        if self.bse.solution is None:
            raise ValueError("No solutions available. Please run estimate_bs() before plotting.")

        if self.bse.y0 is None:
            raise ValueError(
                "No initial conditions available. Please run estimate_bs() before plotting."
            )

        if self.bse.solution.features is None:
            raise ValueError("No features available. Please run estimate_bs() before plotting.")

        if self.bse.solution.labels is None:
            raise ValueError("No labels available. Please run estimate_bs() before plotting.")

        if self.bse.bs_vals is None:
            raise ValueError(
                "No basin stability values available. Please run estimate_bs() before plotting."
            )

        # Extract data from each Solution instance.
        initial_conditions = self.bse.y0.cpu().numpy()

        features_array = self.bse.solution.features.cpu().numpy()

        # ['LC' 'LC' 'FP' 'LC' 'LC' ... ]
        labels = np.array(self.bse.solution.labels)

        plt.figure(figsize=(10, 10))  # type: ignore[misc]

        # 1) Bar plot for basin stability values.
        plt.subplot(2, 2, 1)  # type: ignore[misc]
        bar_labels, values = zip(*self.bse.bs_vals.items(), strict=True)
        plt.bar(bar_labels, values, color=["#ff7f0e", "#1f77b4"])  # type: ignore[misc]
        plt.xticks(bar_labels)  # type: ignore[misc]
        plt.ylabel("Fraction of samples")  # type: ignore[misc]
        plt.title("Basin Stability")  # type: ignore[misc]

        # 2) State space scatter plot: class-labeled initial conditions.
        plt.subplot(2, 2, 2)  # type: ignore[misc]
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = np.where(labels == label)
            plt.scatter(  # type: ignore[misc]
                initial_conditions[idx, 0],
                initial_conditions[idx, 1],
                s=4,
                alpha=0.5,
                label=str(label),
            )
        plt.title("Initial Conditions in State Space")  # type: ignore[misc]
        # TODO: Have custom labels per case
        plt.xlabel("y_1")  # type: ignore[misc]
        plt.ylabel("y_2")  # type: ignore[misc]
        plt.legend(loc="upper left")  # type: ignore[misc]

        # 3) Feature space scatter plot with classifier results.
        plt.subplot(2, 2, 3)  # type: ignore[misc]
        for label in unique_labels:
            idx = np.where(labels == label)
            # Map labels to class names if desired (example mapping below)
            plt.scatter(  # type: ignore[misc]
                features_array[idx, 0], features_array[idx, 1], s=5, alpha=0.5, label=str(label)
            )
        plt.title("Feature Space with Classifier Results")  # type: ignore[misc]
        plt.xlabel("Feature 1")  # type: ignore[misc]
        plt.ylabel("Feature 2")  # type: ignore[misc]
        plt.legend()  # type: ignore[misc]

        # 4) Placeholder for future plotting.
        plt.subplot(2, 2, 4)  # type: ignore[misc]
        plt.title("Future Plot")  # type: ignore[misc]

        plt.tight_layout()

        # Save the figure
        # Create results directory if it does not exist
        if self.bse.save_to:
            self.save_plot("bse_results_plot")

        plt.show()  # type: ignore[misc]

    # Plots 2 states over time for the same trajectory in the same space
    def plot_phase(self, x_var: int = 0, y_var: int = 1, z_var: int | None = None):
        """
        Plot trajectories for the template initial conditions in 2D or 3D phase space.
        """
        if not isinstance(self.bse.cluster_classifier, ClassifierPredictor):
            raise ValueError(
                "plot_phase requires a ClassifierPredictor with template initial conditions."
            )

        # Use classifier's solver if available, otherwise use main solver
        solver = self.bse.cluster_classifier.solver or self.bse.solver  # type: ignore[misc]

        # Convert template_y0 list to tensor on solver's device
        template_tensor = torch.tensor(
            self.bse.cluster_classifier.template_y0,  # type: ignore[misc]
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
                zip(self.bse.cluster_classifier.labels, trajectories.permute(1, 0, 2), strict=True)  # type: ignore[arg-type]
            ):
                ax.plot(traj[:, x_var], traj[:, y_var], label=str(label))  # type: ignore[misc]
            ax.set_xlabel(f"State {x_var}")  # type: ignore[misc]
            ax.set_ylabel(f"State {y_var}")  # type: ignore[misc]
            ax.set_title("2D Phase Plot")  # type: ignore[misc]
        else:
            ax = fig.add_subplot(111, projection="3d")  # type: ignore[assignment]
            for _i, (label, traj) in enumerate(
                zip(self.bse.cluster_classifier.labels, trajectories.permute(1, 0, 2), strict=True)  # type: ignore[arg-type]
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

        plt.show()  # type: ignore[misc]

    # Plots over time
    def plot_templates(self, plotted_var: int, time_span: tuple[float, float] | None = None):
        """
        Plot trajectories for the template initial conditions.

        Args:
            plotted_var (int): Index of the variable to plot
            time_span (tuple, optional): Time range to plot (t_start, t_end)
        """
        if not isinstance(self.bse.cluster_classifier, ClassifierPredictor):
            raise ValueError(
                "plot_templates requires a ClassifierPredictor with template initial conditions."
            )

        # Use classifier's solver if available, otherwise use main solver
        solver = self.bse.cluster_classifier.solver or self.bse.solver  # type: ignore[misc]

        # Convert template_y0 list to tensor on solver's device
        template_tensor = torch.tensor(
            self.bse.cluster_classifier.template_y0,  # type: ignore[misc]
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
            zip(self.bse.cluster_classifier.labels, y.permute(1, 0, 2), strict=True)  # type: ignore[arg-type]
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

        plt.show()  # type: ignore[misc]
