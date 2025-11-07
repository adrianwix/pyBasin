"""Adaptive Study Basin Stability Estimator."""

import os
from typing import Literal, TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from pybasin.as_basin_stability_estimator import ASBasinStabilityEstimator
from pybasin.utils import generate_filename, resolve_folder

matplotlib.use("Agg")


class AdaptiveStudyParams(TypedDict):
    # TODO: Delete mode
    mode: Literal["hyper_parameter", "model_parameter"]
    adaptative_parameter_values: list
    adaptative_parameter_name: str


class ASPlotter:
    """
    Adaptive Study Basin Stability Estimator.
    """

    def __init__(self, as_bse: ASBasinStabilityEstimator):
        """
        Initialize the Plotter with a ASBasinStabilityEstimator instance.

        :param bse: An instance of ASBasinStabilityEstimator.
        """
        self.as_bse = as_bse

    def save_plot(self, plot_name: str):
        if self.as_bse.save_to is None:
            raise ValueError("save_to is not defined.")

        full_folder = resolve_folder(self.as_bse.save_to)
        file_name = generate_filename(plot_name, "png")
        full_path = os.path.join(full_folder, file_name)

        print("Saving plots to: ", full_path)
        plt.savefig(full_path, dpi=300)

    def plot_basin_stability_variation(self, interval: Literal["linear", "log"] = "linear"):
        """
        Plot all basin stability values against parameter variation in a single plot.

        :param interval: Indicates whether the x-axis should use a linear or logarithmic scale.
                         - 'linear': Default linear scale.
                         - 'log':    Logarithmic scale, e.g., when using 2 * np.logspace(...).
        """
        if not self.as_bse.parameter_values or not self.as_bse.basin_stabilities:
            raise ValueError("No results available. Run estimate_as_bs first.")

        # Collect all unique labels across all basin_stabilities
        labels = set()
        for bs_dict in self.as_bse.basin_stabilities:
            labels.update(bs_dict.keys())
        labels = list(labels)  # Convert to list for consistent ordering

        # Convert list of dictionaries to arrays for plotting
        bs_values = {label: [] for label in labels}

        # Reorganize data by label
        for bs_dict in self.as_bse.basin_stabilities:
            for label in labels:
                # Use 0 if label is missing
                bs_values[label].append(bs_dict.get(label, 0))

        # Create single plot with all labels
        plt.figure(figsize=(10, 6))

        # Set x-axis scale if needed
        if interval == "log":
            plt.xscale("log")

        # Plot each label's data
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # Different colors

        for i, label in enumerate(labels):
            plt.plot(
                self.as_bse.parameter_values,
                bs_values[label],
                "o-",  # Use consistent marker 'o' for all
                color=colors[i % len(colors)],
                label=f"State {label}",
                markersize=8,
                linewidth=2,
                alpha=0.8,
            )

        plt.xlabel(self.as_bse.as_params["adaptative_parameter_name"].split(".")[-1])
        plt.ylabel("Basin Stability")
        plt.title("Basin Stability vs Parameter Variation")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save plot
        if self.as_bse.save_to:
            self.save_plot("basin_stability_variation")

        plt.show()

    def get_amplitudes(self, solution, dof, n_clusters):
        """
        Extract amplitudes and compute differences via k-means clustering.
        Assumes solution.bifurcation_amplitudes has been extracted using
        extract_amplitudes (from utils.py) and might be a torch.Tensor.

        Args:
            solution: Solution object with attribute bifurcation_amplitudes.
            dof: List of indices for degrees of freedom to analyze.
            n_clusters: Number of clusters for k-means.

        Returns:
            centers: Array of cluster centroids (shape: n_clusters x len(dof)).
            diffs: Mean absolute differences (shape: n_clusters x len(dof)).
        """
        # Extract the relevant amplitudes.
        temp = solution.bifurcation_amplitudes[:, dof]

        # If temp is a torch.Tensor, convert it to a NumPy array.
        temp = temp.detach().cpu().numpy() if hasattr(temp, "detach") else np.asarray(temp)

        # k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(temp)
        centers = kmeans.cluster_centers_

        n_dofs = len(dof)
        diffs = np.zeros((n_clusters, n_dofs))
        for i in range(n_clusters):
            for j in range(n_dofs):
                if np.any(labels == i):
                    diffs[i, j] = np.mean(np.abs(temp[labels == i, j] - centers[i, j]))
                else:
                    diffs[i, j] = 0
        return centers, diffs

    def plot_bifurcation_diagram(self, dof: list[int]):
        """
        Plot bifurcation diagram showing attractor locations over parameter variation.

        For each parameter value, the method extracts the bifurcation amplitudes
        (i.e. solution.bifurcation_amplitudes), selects the desired DOFs, applies
        k-means clustering and then plots the cluster centers as a function of
        the parameter.

        Args:
            dof: List of indices of the state variables (DOFs) to plot.
        """
        if not self.as_bse.parameter_values or not self.as_bse.results:
            raise ValueError("No results available. Run estimate_as_bs first.")

        # Use the state_dim as the number of clusters.
        n_clusters = self.as_bse.sampler.state_dim
        n_dofs = len(dof)
        n_par_var = len(self.as_bse.results)

        # Pre-allocate storage for cluster centers and errors
        amplitudes = np.zeros((n_clusters, n_dofs, n_par_var))
        errors = np.zeros((n_clusters, n_dofs, n_par_var))

        # Process each parameter variation
        for idx, result in enumerate(self.as_bse.results):
            # Get bifurcation amplitudes directly from result dict
            bifurcation_amplitudes = result["bifurcation_amplitudes"]
            if bifurcation_amplitudes is None:
                raise ValueError(
                    f"No bifurcation amplitudes found for parameter value {result['param_value']}"
                )

            # Create a temporary object to hold the amplitudes
            class TempSolution:
                def __init__(self, amps):
                    self.bifurcation_amplitudes = amps

            solution = TempSolution(bifurcation_amplitudes)
            centers, diffs = self.get_amplitudes(solution, dof, n_clusters)
            amplitudes[:, :, idx] = centers
            errors[:, :, idx] = diffs

        # Create subplots for each requested DOF
        fig, axes = plt.subplots(1, n_dofs, figsize=(5 * n_dofs, 4))
        if n_dofs == 1:
            axes = [axes]

        # Set of colors for clusters; extend the list if needed.
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        # Plot the cluster centers vs parameter value for each DOF
        for j in range(n_dofs):
            ax = axes[j]
            for i in range(n_clusters):
                ax.plot(
                    self.as_bse.parameter_values,
                    amplitudes[i, j, :],
                    "o-",  # marker plus line
                    markersize=8,
                    color=colors[i % len(colors)],
                    label=f"Cluster {i + 1}",
                )
            ax.set_xlabel(self.as_bse.as_params["adaptative_parameter_name"].split(".")[-1])
            ax.set_ylabel(f"Amplitude state {dof[j]}")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend()  # Display legend for each subplot

        # Link y-axis limits across subplots
        y_min = min(ax.get_ylim()[0] for ax in axes)
        y_max = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(y_min, y_max)

        plt.suptitle("Bifurcation Diagram")
        plt.tight_layout()

        # Save plot if a save path is set
        if self.as_bse.save_to:
            self.save_plot("bifurcation_diagram")

        plt.show()
