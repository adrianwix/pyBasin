"""Adaptive Study Basin Stability Estimator."""

import json
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


class ASBasinStabilityEstimator:
    """
    Adaptive Study Basin Stability Estimator.
    """

    def __init__(
        self,
        name: str,
        N: int,
        ode_system: ODESystem,
        sampler: Sampler,
        solver: Solver,
        feature_extractor: FeatureExtractor,
        cluster_classifier: ClusterClassifier,
        as_params: AdaptiveStudyParams,
        save_to: Optional[str] = "results"
    ):
        """
        Initialize the BasinStabilityEstimator.

        :param name: Name of the case study. Used for saving results.
        :param N: Number of initial conditions (samples) to generate.
        :param steady_state_time: Time after which steady-state features are extracted.
        :param ode_system: The ODE system model.
        :param sampler: The Sampler object to generate initial conditions.
        :param solver: The Solver object to integrate the ODE system.
        :param cluster_classifier: The ClusterClassifier object to assign labels.
        :param as_params: The AdaptiveStudyParams object to vary the parameter.
        :param save_to: The folder where results will be saved.
        """
        self.name = name
        self.N = N
        self.ode_system = ode_system
        self.sampler = sampler
        self.solver = solver
        self.feature_extractor = feature_extractor
        self.cluster_classifier = cluster_classifier
        self.as_params = as_params
        self.save_to = save_to

        # Add storage for parameter study results
        self.parameter_values = []
        self.basin_stabilities = []

    def estimate_as_bs(self) -> Dict[int, float]:
        """Returns list of basin stability values for each parameter value"""
        self.parameter_values = []
        self.basin_stabilities = []

        print(
            f"\nEstimating Basin Stability for parameter: {self.as_params['adaptative_parameter_name']}")

        print(
            f"\nParameter values: {self.as_params['adaptative_parameter_values']}")

        for param_value in self.as_params["adaptative_parameter_values"]:
            # Update parameter using eval
            assignment = f"{self.as_params['adaptative_parameter_name']} = {
                param_value}"

            eval(compile(assignment, '<string>', 'exec'), {
                "ode_system": self.ode_system,
                "sampler": self.sampler,
                "solver": self.solver,
                "feature_extractor": self.feature_extractor,
                "cluster_classifier": self.cluster_classifier,
            })

            print(
                f"\nCurrent {self.as_params['adaptative_parameter_name']} value: {param_value}")

            bse = BasinStabilityEstimator(
                # Add parameter value to name
                name=f"{self.name}_{param_value}",
                N=self.N,
                ode_system=self.ode_system,
                sampler=self.sampler,
                solver=self.solver,
                feature_extractor=self.feature_extractor,
                cluster_classifier=self.cluster_classifier
            )

            basin_stability = bse.estimate_bs()

            # Store results
            self.parameter_values.append(param_value)
            self.basin_stabilities.append(basin_stability)

            print(f"Basin Stability: {
                  basin_stability} for parameter value {param_value}")

        return self.parameter_values, self.basin_stabilities

    def plot_basin_stability_variation(self):
        """Plot all basin stability values against parameter variation in a single plot"""
        if not self.parameter_values or not self.basin_stabilities:
            raise ValueError("No results available. Run estimate_as_bs first.")

        # Collect all unique labels across all basin_stabilities
        labels = set()
        for bs_dict in self.basin_stabilities:
            labels.update(bs_dict.keys())
        labels = list(labels)  # Convert to list for consistent ordering

        # Convert list of dictionaries to arrays for plotting
        bs_values = {label: [] for label in labels}

        # Reorganize data by label
        for bs_dict in self.basin_stabilities:
            for label in labels:
                # Use 0 if label is missing
                bs_values[label].append(bs_dict.get(label, 0))

        # Create single plot with all labels
        plt.figure(figsize=(10, 6))

        # Plot each label's data
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
                  '#d62728', '#9467bd']  # Different colors

        for i, label in enumerate(labels):
            plt.plot(self.parameter_values, bs_values[label],
                     'o-',  # Use consistent marker 'o' for all
                     color=colors[i % len(colors)],
                     label=f'State {label}',
                     markersize=8,
                     linewidth=2,
                     alpha=0.8)

        plt.xlabel(self.as_params['adaptative_parameter_name'].split(
            '.')[-1])  # Use last part of parameter name
        plt.ylabel('Basin Stability')
        plt.title('Basin Stability vs Parameter Variation')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save plot
        results_dir = f"results_{self.name}"
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'basin_stability_variation.png'),
                    dpi=300, bbox_inches='tight')

        plt.show()

    def plot_bifurcation_diagram(self, dof: list[int]):
        """
        Plot bifurcation diagram showing how attractor amplitudes vary with parameter.

        Args:
            dof (list[int]): Indices of degrees of freedom to plot
        """
        if not self.parameter_values or not self.basin_stabilities:
            raise ValueError("No results available. Run estimate_as_bs first.")

        n_dofs = len(dof)
        n_clusters = len(self.cluster_classifier.labels)
        n_params = len(self.parameter_values)

        # Storage for amplitudes and errors
        amplitudes = np.full((n_clusters, n_dofs, n_params), np.nan)
        errors = np.full((n_clusters, n_dofs, n_params), np.nan)

        # Process each parameter value
        for idx_p, param_value in enumerate(self.parameter_values):
            # Get solutions for current parameter
            solutions = self.solutions[idx_p * self.N:(idx_p + 1) * self.N]

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
                ax.plot(self.parameter_values,
                        amplitudes[idx_c, idx_d, :],
                        'k.',
                        markersize=8)

            ax.set_xlabel(
                self.as_params['adaptative_parameter_name'].split('.')[-1])
            ax.set_ylabel(f'Amplitude state {dof[idx_d]}')
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Save plot
        results_dir = f"results_{self.name}"
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'bifurcation_diagram.png'),
                    dpi=300, bbox_inches='tight')

        plt.show()

    # TODO: Plot plot_bs_parameter_study as plot_bif_diagram

    def plots(self):
        pass

    def save(self):
        """
        Save the basin stability results to a JSON file.
        Handles numpy arrays and Solution objects by converting them to standard Python types.
        """
        if self.basin_stabilities is None:
            raise ValueError(
                "No results to save. Please run estimate_as_bs() first.")

        if self.save_to is None:
            raise ValueError(
                "No path to save the results was specified.")

        full_folder = resolve_folder(self.save_to)
        file_name = generate_filename(
            'adaptative_params_basin_stability_results', 'json')
        full_path = os.path.join(full_folder, file_name)

        def format_ode_system(ode_str: str) -> list:
            lines = ode_str.strip().split('\n')
            formatted_lines = [' '.join(line.split()) for line in lines]
            return formatted_lines

        region_of_interest = " X ".join(
            [f"[{min_val}, {max_val}]" for min_val, max_val in zip(
                self.sampler.min_limits, self.sampler.max_limits)]
        )

        adaptative_parameter_study = [
            {"param_value": param_value, "basin_of_attraction": bs}
            for param_value, bs in zip(self.parameter_values, self.basin_stabilities)]

        results = {
            "studied_parameters": self.as_params['adaptative_parameter_name'],
            "adaptative_parameter_study": adaptative_parameter_study,
            "region_of_interest": region_of_interest,
            "sampling_points": self.N,
            "sampling_method": self.sampler.__class__.__name__,
            "solver": self.solver.__class__.__name__,
            "cluster_classifier": self.cluster_classifier.__class__.__name__,
            "ode_system": format_ode_system(self.ode_system.get_str()),
        }

        with open(full_path, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)

        print(f"Results saved to {full_path}")
