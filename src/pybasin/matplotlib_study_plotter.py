"""Parameter Study Basin Stability Plotter."""

import logging
import os
from collections import defaultdict
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans

from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.utils import generate_filename, resolve_folder

logger = logging.getLogger(__name__)


class MatplotlibStudyPlotter:
    """
    Matplotlib-based plotter for parameter study basin stability results.

    Supports multi-parameter studies by grouping results along one chosen parameter
    (x-axis) while producing separate curves for each combination of the remaining
    parameters. When ``parameters`` is not passed, one figure is produced per
    studied parameter.

    :ivar bs_study: BasinStabilityStudy instance with computed results.
    """

    def __init__(self, bs_study: BasinStabilityStudy):
        """
        Initialize the plotter with a BasinStabilityStudy instance.

        :param bs_study: An instance of BasinStabilityStudy.
        """
        self.bs_study = bs_study
        self._pending_figures: list[tuple[str, Figure]] = []

    def save(self, dpi: int = 300) -> None:
        """
        Save all pending figures to the save_to directory.

        Figures are tracked when plot methods create new figures. Call this
        after plotting to save all figures at once.

        :param dpi: Resolution for saved images.
        :raises ValueError: If ``bs_study.save_to`` is not set or no figures pending.
        """
        if self.bs_study.save_to is None:
            raise ValueError("bs_study.save_to is not defined. Set it before calling save().")

        if not self._pending_figures:
            raise ValueError("No figures to save. Call a plot method first.")

        full_folder = resolve_folder(self.bs_study.save_to)

        for name, fig in self._pending_figures:
            file_name = generate_filename(name, "png")
            full_path = os.path.join(full_folder, file_name)
            logger.info("Saving plot to: %s", full_path)
            fig.savefig(full_path, dpi=dpi)  # type: ignore[misc]

        self._pending_figures.clear()

    def show(self) -> None:
        """
        Display all matplotlib figures.

        Convenience wrapper around ``plt.show()`` so users don't need to
        import matplotlib separately.
        """
        plt.show()  # type: ignore[misc]

    def _group_by_parameter(self, param_name: str) -> dict[tuple[tuple[str, Any], ...], list[int]]:
        """Group study result indices by the values of all parameters except ``param_name``.

        Within each group the indices are sorted by ``param_name``'s value so they
        can be plotted as a line.

        :param param_name: The parameter whose values form the x-axis.
        :return: Mapping from a tuple of (other_param, value) pairs to sorted result indices.
        """
        other_params: list[str] = [
            p for p in self.bs_study.studied_parameter_names if p != param_name
        ]

        groups: dict[tuple[tuple[str, Any], ...], list[int]] = defaultdict(list)
        for i, r in enumerate(self.bs_study.results):
            sl = r["study_label"]
            group_key = tuple((p, sl[p]) for p in other_params) if other_params else ()
            groups[group_key].append(i)

        for group_key in groups:
            groups[group_key].sort(
                key=lambda i: self.bs_study.results[i]["study_label"][param_name]
            )

        return dict(groups)

    def _resolve_parameters(self, parameters: list[str] | None) -> list[str]:
        """Resolve which parameters to iterate over.

        :param parameters: Explicit list or None for all.
        :return: List of parameter names.
        :raises ValueError: If a name is not among the studied parameters.
        """
        all_names = self.bs_study.studied_parameter_names
        if parameters is None:
            return all_names
        for p in parameters:
            if p not in all_names:
                raise ValueError(f"Parameter '{p}' not found. Studied parameters: {all_names}")
        return parameters

    def _get_attractor_labels(self) -> list[str]:
        """Collect all unique attractor labels across every run, sorted."""
        labels_set: set[str] = set()
        for r in self.bs_study.results:
            labels_set.update(r["basin_stability"].keys())
        return sorted(labels_set)

    @staticmethod
    def _generate_colors(n: int) -> list[Any]:
        """Generate ``n`` visually distinct colors from a matplotlib colormap.

        Uses ``tab10`` for up to 10 colors, ``tab20`` for up to 20, and
        ``hsv`` for larger counts so that colors never repeat within a figure.

        :param n: Number of distinct colors required.
        :return: List of RGBA color tuples.
        """
        if n <= 1:
            return [plt.cm.tab10(0)]  # type: ignore[misc]
        if n <= 10:
            cmap = plt.cm.tab10  # type: ignore[misc]
        elif n <= 20:
            cmap = plt.cm.tab20  # type: ignore[misc]
        else:
            cmap = plt.cm.hsv  # type: ignore[misc]
        return [cmap(i / n) for i in range(n)]

    def plot_basin_stability_variation(
        self,
        interval: Literal["linear", "log"] = "linear",
        parameters: list[str] | None = None,
    ) -> list[Figure]:
        """Plot basin stability values against parameter variation.

        Produces one figure per parameter. For multi-parameter studies,
        results are grouped by the other parameters and each group becomes
        a separate set of lines in the figure.

        :param interval: x-axis scale — ``'linear'`` or ``'log'``.
        :param parameters: Which studied parameters to plot. ``None`` plots all.
        :return: List of matplotlib Figure objects (one per parameter).
        """
        if not self.bs_study.results:
            raise ValueError("No results available. Run study first.")

        params_to_plot = self._resolve_parameters(parameters)
        attractor_labels = self._get_attractor_labels()

        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+"]
        linestyles = ["-", "--", ":", "-."]
        all_groups = {p: self._group_by_parameter(p) for p in params_to_plot}
        max_n_groups = max(len(g) for g in all_groups.values()) if all_groups else 1
        colors = self._generate_colors(max_n_groups)
        figures: list[Figure] = []

        for param_name in params_to_plot:
            groups = all_groups[param_name]

            fig = plt.figure(figsize=(10, 6))  # type: ignore[misc]

            if interval == "log":
                plt.xscale("log")  # type: ignore[misc]

            for g_idx, (_group_key, indices) in enumerate(groups.items()):
                x_values = [self.bs_study.results[i]["study_label"][param_name] for i in indices]

                for a_idx, attractor in enumerate(attractor_labels):
                    y_values = [
                        self.bs_study.results[i]["basin_stability"].get(attractor, 0)
                        for i in indices
                    ]
                    plt.plot(  # type: ignore[misc]
                        x_values,
                        y_values,
                        marker=markers[a_idx // len(linestyles) % len(markers)],
                        linestyle=linestyles[a_idx % len(linestyles)],
                        color=colors[g_idx],
                        markersize=8,
                        linewidth=2,
                        alpha=0.8,
                    )

            attractor_handles = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=2,
                    linestyle=linestyles[a_idx % len(linestyles)],
                    marker=markers[a_idx // len(linestyles) % len(markers)],
                    markersize=8,
                    label=attractor,
                )
                for a_idx, attractor in enumerate(attractor_labels)
            ]
            group_handles = [
                Line2D(
                    [0],
                    [0],
                    color=colors[g_idx],
                    linewidth=4,
                    label=", ".join(f"{k}={v}" for k, v in group_key) if group_key else "all",
                )
                for g_idx, group_key in enumerate(groups.keys())
            ]
            legend1 = plt.legend(  # type: ignore[misc]
                handles=attractor_handles,
                title="Attractor",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                handlelength=5,
            )
            plt.gca().add_artist(legend1)  # type: ignore[misc]
            plt.legend(  # type: ignore[misc]
                handles=group_handles,
                title="Group",
                bbox_to_anchor=(1.05, 0.5),
                loc="upper left",
            )
            plt.xlabel(param_name)  # type: ignore[misc]
            plt.ylabel("Basin Stability")  # type: ignore[misc]
            plt.title(f"Basin Stability vs {param_name}")  # type: ignore[misc]
            plt.grid(True, linestyle="--", alpha=0.7)  # type: ignore[misc]
            plt.tight_layout()

            self._pending_figures.append((f"basin_stability_variation_{param_name}", fig))
            figures.append(fig)

        return figures

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

    def plot_bifurcation_diagram(
        self,
        dof: list[int],
        parameters: list[str] | None = None,
    ) -> list[Figure]:
        """Plot bifurcation diagrams showing attractor locations over parameter variation.

        Produces one figure per parameter. For multi-parameter studies,
        results are grouped by the other parameters and each group becomes
        a separate set of curves.

        :param dof: List of indices of the state variables (DOFs) to plot.
        :param parameters: Which studied parameters to plot. ``None`` plots all.
        :return: List of matplotlib Figure objects (one per parameter).
        """
        if not self.bs_study.results:
            raise ValueError("No results available. Run study first.")

        params_to_plot = self._resolve_parameters(parameters)
        n_clusters = self.bs_study.sampler.state_dim
        n_dofs = len(dof)

        colors = self._generate_colors(n_clusters)
        figures: list[Figure] = []

        for param_name in params_to_plot:
            groups = self._group_by_parameter(param_name)

            fig, axes = plt.subplots(1, n_dofs, figsize=(5 * n_dofs, 4))  # type: ignore[misc]
            if n_dofs == 1:
                axes = [axes]

            for group_key, indices in groups.items():
                x_values = [self.bs_study.results[i]["study_label"][param_name] for i in indices]

                n_par_var = len(indices)
                amplitudes = np.zeros((n_clusters, n_dofs, n_par_var))
                errors = np.zeros((n_clusters, n_dofs, n_par_var))

                for pos, result_idx in enumerate(indices):
                    result = self.bs_study.results[result_idx]
                    bifurcation_amplitudes = result["bifurcation_amplitudes"]
                    if bifurcation_amplitudes is None:
                        study_label = self.bs_study.results[result_idx]["study_label"]
                        raise ValueError(
                            f"No bifurcation amplitudes found for parameter combination {study_label}"
                        )

                    class TempSolution:
                        def __init__(self, amps: Any):
                            self.bifurcation_amplitudes = amps

                    solution = TempSolution(bifurcation_amplitudes)
                    centers, diffs = self._get_amplitudes(solution, dof, n_clusters)
                    amplitudes[:, :, pos] = centers
                    errors[:, :, pos] = diffs

                group_suffix = ""
                if group_key:
                    group_suffix = " (" + ", ".join(f"{k}={v}" for k, v in group_key) + ")"

                for j in range(n_dofs):
                    ax = axes[j]
                    for i in range(n_clusters):
                        ax.plot(  # type: ignore[misc]
                            x_values,
                            amplitudes[i, j, :],
                            "o-",
                            markersize=8,
                            color=colors[i % n_clusters],
                            label=f"Cluster {i + 1}{group_suffix}",
                        )

            for j in range(n_dofs):
                axes[j].set_xlabel(param_name)  # type: ignore[misc]
                axes[j].set_ylabel(f"Amplitude state {dof[j]}")  # type: ignore[misc]
                axes[j].grid(True, linestyle="--", alpha=0.7)  # type: ignore[misc]
                axes[j].legend()  # type: ignore[misc]

            y_min = min(ax.get_ylim()[0] for ax in axes)  # type: ignore[misc]
            y_max = max(ax.get_ylim()[1] for ax in axes)  # type: ignore[misc]
            for ax in axes:
                ax.set_ylim(y_min, y_max)  # type: ignore[misc]

            plt.suptitle(f"Bifurcation Diagram ({param_name})")  # type: ignore[misc]
            plt.tight_layout()

            self._pending_figures.append((f"bifurcation_diagram_{param_name}", fig))
            figures.append(fig)

        return figures
