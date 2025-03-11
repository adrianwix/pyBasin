"""Adaptive Study Basin Stability Estimator."""

import json
from pybasin.ASBasinStabilityEstimator import ASBasinStabilityEstimator
from pybasin.utils import NumpyEncoder, generate_filename, resolve_folder
from pybasin.Solver import Solver
from pybasin.Sampler import Sampler
from typing import Dict, Optional
from pybasin.ODESystem import ODESystem
from pybasin.FeatureExtractor import FeatureExtractor
from pybasin.ClusterClassifier import ClusterClassifier
from pybasin.BasinStabilityEstimator import BasinStabilityEstimator
from sklearn.cluster import KMeans
from typing import Literal, Dict, TypedDict, Union
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


class AdaptiveStudyParams(TypedDict):
    # TODO: Delete mode
    mode: Literal['hyper_parameter', 'model_parameter']
    adaptative_parameter_values: list
    adaptative_parameter_name: str


class ASPlotter:
    """
    Adaptive Study Basin Stability Estimator.
    """

    def __init__(self, as_bse: ASBasinStabilityEstimator):
        """
        Initialize the Plotter with a BasinStabilityEstimator instance.

        :param bse: An instance of BasinStabilityEstimator.
        """
        self.as_bse = as_bse

    def save_plot(self, plot_name: str):
        full_folder = resolve_folder(self.as_bse.save_to)
        file_name = generate_filename(plot_name, 'png')
        full_path = os.path.join(full_folder, file_name)

        print("Saving plots to: ", full_path)
        plt.savefig(full_path, dpi=300)

    def plot_basin_stability_variation(self, interval: Literal['linear', 'log'] = 'linear'):
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
        if interval == 'log':
            plt.xscale('log')

        # Plot each label's data
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
                  '#d62728', '#9467bd']  # Different colors

        for i, label in enumerate(labels):
            plt.plot(self.as_bse.parameter_values, bs_values[label],
                     'o-',  # Use consistent marker 'o' for all
                     color=colors[i % len(colors)],
                     label=f'State {label}',
                     markersize=8,
                     linewidth=2,
                     alpha=0.8)

        plt.xlabel(
            self.as_bse.as_params['adaptative_parameter_name'].split('.')[-1])
        plt.ylabel('Basin Stability')
        plt.title('Basin Stability vs Parameter Variation')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save plot
        if self.as_bse.save_to:
            self.save_plot('basin_stability_variation')

        plt.show()

    def plot_bifurcation_diagram(self, dof: list[int]):
        """
        Plot bifurcation diagram showing how attractor amplitudes vary with parameter.

        Args:
            dof (list[int]): Indices of degrees of freedom to plot
        """
        if not self.as_bse.parameter_values or not self.as_bse.basin_stabilities:
            raise ValueError("No results available. Run estimate_as_bs first.")

        n_dofs = len(dof)
        n_clusters = len(self.as_bse.cluster_classifier.labels)
        n_params = len(self.as_bse.parameter_values)

        # Storage for amplitudes and errors
        amplitudes = np.full((n_clusters, n_dofs, n_params), np.nan)
        errors = np.full((n_clusters, n_dofs, n_params), np.nan)

        # Process each parameter value
        for idx_p, param_value in enumerate(self.as_bse.parameter_values):
            # Get solutions for current parameter
            solutions = self.as_bse.solutions[idx_p *
                                              self.as_bse.N:(idx_p + 1) * self.as_bse.N]

            # Extract steady state values for requested DOFs
            steady_states = np.array([sol.y[-1, dof] for sol in solutions])

            # Perform k-means clustering on steady states
            kmeans = KMeans(n_clusters=n_clusters)
            idx = kmeans.fit_predict(steady_states)
            centers = kmeans.cluster_centers_

            # Store amplitudes (cluster centers)
            amplitudes[:, :, idx_p] = centers

            # Compute errors (mean absolute deviation from center)
            for idx_d in range(n_dofs):
                for idx_c in range(n_clusters):
                    cluster_points = steady_states[idx == idx_c, idx_d]
                    if len(cluster_points) > 0:
                        errors[idx_c, idx_d, idx_p] = np.mean(
                            np.abs(cluster_points - centers[idx_c, idx_d]))

        # Create plot
        fig, axes = plt.subplots(
            1, n_dofs, figsize=(5*n_dofs, 4), squeeze=False)

        for idx_d in range(n_dofs):
            ax = axes[0, idx_d]
            for idx_c in range(n_clusters):
                ax.plot(self.as_bse.parameter_values,
                        amplitudes[idx_c, idx_d, :],
                        'k.',
                        markersize=8)

            ax.set_xlabel(
                self.as_bse.as_params['adaptative_parameter_name'].split('.')[-1])
            ax.set_ylabel(f'Amplitude state {dof[idx_d]}')
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Save plot
        if (self.as_bse.save_to):
            self.save_plot('bifurcation_diagram')

        plt.show()
