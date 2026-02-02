"""Adaptive Study Basin Stability Estimator."""

import logging
import os
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.utils import generate_filename, resolve_folder

logger = logging.getLogger(__name__)


class MatplotlibStudyPlotter:
    """
    Matplotlib-based plotter for adaptive study basin stability results.

    :ivar as_bse: BasinStabilityStudy instance with computed results.
    """

    def __init__(self, as_bse: BasinStabilityStudy):
        """
        Initialize the plotter with an BasinStabilityStudy instance.

        :param as_bse: An instance of BasinStabilityStudy.
        """
        self.as_bse = as_bse

    def save_plot(self, plot_name: str):
        if self.as_bse.save_to is None:
            raise ValueError("save_to is not defined.")

        full_folder = resolve_folder(self.as_bse.save_to)
        file_name = generate_filename(plot_name, "png")
        full_path = os.path.join(full_folder, file_name)

        logger.info("Saving plots to: %s", full_path)
        plt.savefig(full_path, dpi=300)  # type: ignore[func-returns-value]

    def plot_basin_stability_variation(
        self, interval: Literal["linear", "log"] = "linear", show: bool = True
    ):
        """
        Plot all basin stability values against parameter variation in a single plot.

        :param interval: Indicates whether the x-axis should use a linear or logarithmic
            scale. Options:

            - 'linear': Default linear scale.
            - 'log': Logarithmic scale, e.g., when using ``2 * np.logspace(...)``.
        :param show: Whether to display the plot. If False, returns the figure without showing.
        :return: The matplotlib Figure object.
        """
        if not self.as_bse.parameter_values or not self.as_bse.basin_stabilities:
            raise ValueError("No results available. Run estimate_as_bs first.")

        # Collect all unique labels across all basin_stabilities
        labels_set: set[str] = set()
        for bs_dict in self.as_bse.basin_stabilities:
            labels_set.update(bs_dict.keys())
        labels: list[str] = list(labels_set)  # Convert to list for consistent ordering

        # Convert list of dictionaries to arrays for plotting
        bs_values: dict[str, list[float]] = {label: [] for label in labels}

        # Reorganize data by label
        for bs_dict in self.as_bse.basin_stabilities:
            for label in labels:
                # Use 0 if label is missing
                bs_values[label].append(bs_dict.get(label, 0))

        # Create single plot with all labels
        fig = plt.figure(figsize=(10, 6))  # type: ignore[misc]

        # Set x-axis scale if needed
        if interval == "log":
            plt.xscale("log")  # type: ignore[misc]

        # Plot each label's data
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # Different colors

        for i, label in enumerate(labels):
            # Format label: "Unbounded" for unbounded, "State X" for numbered states
            display_label = "Unbounded" if label == "unbounded" else f"State {label}"
            plt.plot(  # type: ignore[misc]
                self.as_bse.parameter_values,
                bs_values[label],
                "o-",  # Use consistent marker 'o' for all
                color=colors[i % len(colors)],
                label=display_label,
                markersize=8,
                linewidth=2,
                alpha=0.8,
            )

        # Get parameter name from first label key
        param_name = next(iter(self.as_bse.labels[0].keys())) if self.as_bse.labels else "Parameter"
        plt.xlabel(param_name)  # type: ignore[misc]
        plt.ylabel("Basin Stability")  # type: ignore[misc]
        plt.title("Basin Stability vs Parameter Variation")  # type: ignore[misc]
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # type: ignore[misc]
        plt.grid(True, linestyle="--", alpha=0.7)  # type: ignore[misc]
        plt.tight_layout()

        # Save plot
        if self.as_bse.save_to:
            self.save_plot("basin_stability_variation")

        if show:
            plt.show()  # type: ignore[misc]

        return fig

    def _get_amplitudes(
        self, solution: Any, dof: list[int], n_clusters: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract amplitudes and compute differences via k-means clustering.

        Assumes solution.bifurcation_amplitudes has been extracted using
        extract_amplitudes (from utils.py) and might be a torch.Tensor.

        :param solution: Solution object with attribute bifurcation_amplitudes.
        :param dof: List of indices for degrees of freedom to analyze.
        :param n_clusters: Number of clusters for k-means.
        :return: Tuple of (centers, diffs) where centers is array of cluster
            centroids (shape: n_clusters x len(dof)) and diffs is mean absolute
            differences (shape: n_clusters x len(dof)).
        """
        temp = solution.bifurcation_amplitudes[:, dof]

        temp_np: np.ndarray = (
            temp.detach().cpu().numpy() if hasattr(temp, "detach") else np.asarray(temp)
        )

        finite_mask = np.all(np.isfinite(temp_np), axis=1)
        temp_np_finite = temp_np[finite_mask]

        if len(temp_np_finite) == 0:
            return np.zeros((n_clusters, len(dof))), np.zeros((n_clusters, len(dof)))

        n_samples_finite = len(temp_np_finite)
        actual_n_clusters = min(n_clusters, n_samples_finite)

        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42)
        labels: np.ndarray = kmeans.fit_predict(temp_np_finite)  # type: ignore[assignment]
        centers_raw = kmeans.cluster_centers_  # type: ignore[assignment]
        centers: np.ndarray = np.asarray(centers_raw)  # type: ignore[arg-type]

        if actual_n_clusters < n_clusters:
            centers_padded = np.zeros((n_clusters, len(dof)))
            centers_padded[:actual_n_clusters] = centers
            centers = centers_padded

        n_dofs = len(dof)
        diffs = np.zeros((n_clusters, n_dofs))
        for i in range(actual_n_clusters):
            for j in range(n_dofs):
                if np.any(labels == i):  # type: ignore[arg-type]
                    diffs[i, j] = np.mean(np.abs(temp_np_finite[labels == i, j] - centers[i, j]))  # type: ignore[arg-type]
        return centers, diffs

    def plot_bifurcation_diagram(self, dof: list[int], show: bool = True):
        """
        Plot bifurcation diagram showing attractor locations over parameter variation.

        For each parameter value, the method extracts the bifurcation amplitudes
        (i.e. solution.bifurcation_amplitudes), selects the desired DOFs, applies
        k-means clustering and then plots the cluster centers as a function of
        the parameter.

        Unbounded trajectories are automatically filtered out before clustering,
        as bifurcation amplitudes are only computed for bounded trajectories.

        :param dof: List of indices of the state variables (DOFs) to plot.
        :param show: Whether to display the plot. If False, returns the figure without showing.
        :return: The matplotlib Figure object.
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

            # Note: bifurcation_amplitudes only contains bounded trajectories
            # (unbounded trajectories are filtered out during basin stability estimation)

            # Create a temporary object to hold the amplitudes
            class TempSolution:
                def __init__(self, amps: Any):
                    self.bifurcation_amplitudes = amps

            solution = TempSolution(bifurcation_amplitudes)
            centers, diffs = self._get_amplitudes(solution, dof, n_clusters)
            amplitudes[:, :, idx] = centers
            errors[:, :, idx] = diffs

        # Create subplots for each requested DOF
        fig, axes = plt.subplots(1, n_dofs, figsize=(5 * n_dofs, 4))  # type: ignore[misc]
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
                ax.plot(  # type: ignore[misc]
                    self.as_bse.parameter_values,
                    amplitudes[i, j, :],
                    "o-",  # marker plus line
                    markersize=8,
                    color=colors[i % len(colors)],
                    label=f"Cluster {i + 1}",
                )
            # Get parameter name from first label key
            param_name = (
                next(iter(self.as_bse.labels[0].keys())) if self.as_bse.labels else "Parameter"
            )
            ax.set_xlabel(param_name)  # type: ignore[misc]
            ax.set_ylabel(f"Amplitude state {dof[j]}")  # type: ignore[misc]
            ax.grid(True, linestyle="--", alpha=0.7)  # type: ignore[misc]
            ax.legend()  # type: ignore[misc]

        # Link y-axis limits across subplots
        y_min = min(ax.get_ylim()[0] for ax in axes)  # type: ignore[misc]
        y_max = max(ax.get_ylim()[1] for ax in axes)  # type: ignore[misc]
        for ax in axes:
            ax.set_ylim(y_min, y_max)  # type: ignore[misc]

        plt.suptitle("Bifurcation Diagram")  # type: ignore[misc]
        plt.tight_layout()

        # Save plot if a save path is set
        if self.as_bse.save_to:
            self.save_plot("bifurcation_diagram")

        if show:
            plt.show()  # type: ignore[misc]

        return fig
