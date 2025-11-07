import os

import matplotlib

matplotlib.use("Agg")  # Set backend before importing pyplot
import numpy as np
from matplotlib import pyplot as plt

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.cluster_classifier import SupervisedClassifier
from pybasin.utils import generate_filename, resolve_folder


class Plotter:
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

        print("Saving plots to: ", full_path)
        plt.savefig(full_path, dpi=300)

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
            raise ValueError("No initial conditions available. Please run estimate_bs() before plotting.")

        if self.bse.solution.features is None:
            raise ValueError("No features available. Please run estimate_bs() before plotting.")

        if self.bse.solution.labels is None:
            raise ValueError("No labels available. Please run estimate_bs() before plotting.")

        if self.bse.bs_vals is None:
            raise ValueError("No basin stability values available. Please run estimate_bs() before plotting.")

        # Extract data from each Solution instance.
        initial_conditions = self.bse.y0.cpu().numpy()

        features_array = self.bse.solution.features.cpu().numpy()

        # ['LC' 'LC' 'FP' 'LC' 'LC' ... ]
        labels = np.array(self.bse.solution.labels)

        plt.figure(figsize=(10, 10))

        # 1) Bar plot for basin stability values.
        plt.subplot(2, 2, 1)
        bar_labels, values = zip(*self.bse.bs_vals.items(), strict=True)
        plt.bar(bar_labels, values, color=["#ff7f0e", "#1f77b4"])
        plt.xticks(bar_labels)
        plt.ylabel("Fraction of samples")
        plt.title("Basin Stability")

        # 2) State space scatter plot: class-labeled initial conditions.
        plt.subplot(2, 2, 2)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = np.where(labels == label)
            plt.scatter(
                initial_conditions[idx, 0], initial_conditions[idx, 1], s=4, alpha=0.5, label=label
            )
        plt.title("Initial Conditions in State Space")
        # TODO: Have custom labels per case
        plt.xlabel("y_1")
        plt.ylabel("y_2")
        plt.legend(loc="upper left")

        # 3) Feature space scatter plot with classifier results.
        plt.subplot(2, 2, 3)
        for label in unique_labels:
            idx = np.where(labels == label)
            # Map labels to class names if desired (example mapping below)
            plt.scatter(features_array[idx, 0], features_array[idx, 1], s=5, alpha=0.5, label=label)
        plt.title("Feature Space with Classifier Results")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()

        # 4) Placeholder for future plotting.
        plt.subplot(2, 2, 4)
        plt.title("Future Plot")

        plt.tight_layout()

        # Save the figure
        # Create results directory if it does not exist
        if self.bse.save_to:
            self.save_plot("bse_results_plot")

        plt.show()

    def plot_phase(self, x_var: int = 0, y_var: int = 1, z_var: int | None = None):
        """
        Plot trajectories for the template initial conditions in 2D or 3D phase space.
        """
        if not isinstance(self.bse.cluster_classifier, SupervisedClassifier):
            raise ValueError("plot_phase requires a SupervisedClassifier with template initial conditions.")

        t, trajectories = self.bse.solver.integrate(
            self.bse.ode_system, self.bse.cluster_classifier.initial_conditions
        )

        fig = plt.figure(figsize=(8, 6))
        if z_var is None:
            ax = fig.add_subplot(111)
            for _i, (label, traj) in enumerate(
                zip(self.bse.cluster_classifier.labels, trajectories.permute(1, 0, 2), strict=True)
            ):
                ax.plot(traj[:, x_var], traj[:, y_var], label=str(label))
            ax.set_xlabel(f"State {x_var}")
            ax.set_ylabel(f"State {y_var}")
            ax.set_title("2D Phase Plot")
        else:
            ax = fig.add_subplot(111, projection="3d")
            for _i, (label, traj) in enumerate(
                zip(self.bse.cluster_classifier.labels, trajectories.permute(1, 0, 2), strict=True)
            ):
                ax.plot(traj[:, x_var], traj[:, y_var], traj[:, z_var], label=str(label))
            ax.set_xlabel(f"State {x_var}")
            ax.set_ylabel(f"State {y_var}")
            ax.set_zlabel(f"State {z_var}")
            ax.set_title("3D Phase Plot")

        plt.legend()
        plt.tight_layout()

        if self.bse.save_to:
            self.save_plot("phase_plot")

        plt.show()

    def plot_templates(self, plotted_var: int, time_span: tuple[float, float] | None = None):
        """
        Plot trajectories for the template initial conditions.

        Args:
            plotted_var (int): Index of the variable to plot
            time_span (tuple, optional): Time range to plot (t_start, t_end)
        """
        if not isinstance(self.bse.cluster_classifier, SupervisedClassifier):
            raise ValueError("plot_templates requires a SupervisedClassifier with template initial conditions.")

        # Get trajectories for template initial conditions
        t, y = self.bse.solver.integrate(
            self.bse.ode_system, self.bse.cluster_classifier.initial_conditions
        )

        plt.figure(figsize=(8, 6))

        # Filter time if specified
        idx = (t >= time_span[0]) & (t <= time_span[1]) if time_span is not None else slice(None)

        # Plot each trajectory
        # Use permute instead of transpose for 3D tensors
        for _i, (label, traj) in enumerate(
            zip(self.bse.cluster_classifier.labels, y.permute(1, 0, 2), strict=True)
        ):
            plt.plot(t[idx], traj[idx, plotted_var], label=f"{label}")

        plt.xlabel("Time")
        plt.ylabel(f"State {plotted_var}")
        plt.title("Template Trajectories")
        plt.legend()
        plt.grid(True)

        # Save plot
        if self.bse.save_to:
            self.save_plot("template_trajectories_plot")

        plt.show()
