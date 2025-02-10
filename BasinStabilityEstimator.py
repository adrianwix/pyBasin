import torch
from torchdiffeq import odeint
from json import JSONEncoder
import json
from Solver import Solver
from Solution import Solution
from Sampler import Sampler
from typing import Dict, Optional
from ODESystem import ODESystem
from FeatureExtractor import FeatureExtractor
from ClusterClassifier import ClusterClassifier
from typing import Optional, Dict, Union
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class BasinStabilityEstimator:
    """
    BasinStabilityEstimator (BSE): Core class for basin stability analysis.

    This class configures the analysis with an ODE system, sampler, and solver,
    and it provides methods to estimate the basin stability (estimate_bs),
    generate diagnostic plots (plots), and save results to file (save).

    Attributes:
        bs_vals (Optional[Dict[int, float]]): Basin stability values (fraction of samples per class).
        Y0 (np.ndarray): Array of initial conditions.
        solution (Solution): Solution instance.
    """
    solution: Solution
    bs_vals: Optional[Dict[int, float]]

    def __init__(
        self,
        name: str,
        N: int,
        ode_system: ODESystem,
        sampler: Sampler,
        solver: Solver,
        feature_extractor: FeatureExtractor,
        cluster_classifier: ClusterClassifier,
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
        """
        self.name = name
        self.N = N
        self.ode_system = ode_system
        self.sampler = sampler
        self.solver = solver
        self.feature_extractor = feature_extractor
        self.cluster_classifier = cluster_classifier

        # Attributes to be populated during estimation
        self.bs_vals: Optional[Dict[int, float]] = None
        self.Y0 = None
        self.solution = None

    def estimate_bs(self) -> Dict[int, float]:
        """
        Estimate basin stability by:
            1. Generating initial conditions using the sampler.
            2. Integrating the ODE system for each sample (in parallel) to produce a Solution.
            3. Extracting features from each Solution.
            4. Clustering/classifying the feature space.
            5. Computing the fraction of samples in each basin.

        This method sets:
            - self.Y0
            - self.solution
            - self.bs_vals

        :return: A dictionary of basin stability values per class.
        """
        # Step 1: Generate initial conditions.
        self.Y0 = self.sampler.sample(self.N)

        # Step 2/3: Integrate and extract features (no parallelization needed).
        t, y = self.solver.integrate(self.ode_system, self.Y0)

        # Initialize solutions list
        self.solution = Solution(
            initial_condition=self.Y0,
            time=t,
            y=y
        )

        # Build the features array from the Solution instances.
        features = self.feature_extractor.extract_features(self.solution)

        self.solution.set_features(features)

        # Step 4: Perform clustering/classification.
        if self.cluster_classifier.type == 'supervised':
            self.cluster_classifier.fit(
                solver=self.solver,
                ode_system=self.ode_system,
                feature_extractor=self.feature_extractor)

        labels = self.cluster_classifier.get_labels(features)

        self.solution.set_labels(labels)

        # Step 5: Compute basin stability as the fraction of samples per class.
        # Initialize with zeros for all possible labels
        self.bs_vals = {label: 0.0 for label in labels}

        # Update with actual fractions where they exist
        unique_labels, counts = np.unique(labels, return_counts=True)
        fractions = counts / float(self.N)

        for label, fraction in zip(unique_labels, fractions):
            self.bs_vals[label] = fraction

        return self.bs_vals

    def plots(self):
        """
        Generate diagnostic plots using the data stored in self.solution:
            1. A bar plot of basin stability values.
            2. A scatter plot of initial conditions (state space).
            3. A scatter plot of the feature space with classifier results.
            4. A placeholder plot for future use.
        """
        if self.solution is None:
            raise ValueError(
                "No solutions available. Please run estimate_bs() before plotting.")

        # Extract data from each Solution instance.
        initial_conditions = self.solution.initial_condition.cpu().numpy()

        features_array = self.solution.features.cpu().numpy()

        # ['LC' 'LC' 'FP' 'LC' 'LC' ... ]
        labels = np.array(self.solution.labels)

        plt.figure(figsize=(10, 6))

        # 1) Bar plot for basin stability values.
        plt.subplot(2, 2, 1)
        bar_labels, values = zip(*self.bs_vals.items())
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
                initial_conditions[idx, 0],
                initial_conditions[idx, 1],
                s=5,
                alpha=0.5,
                label=label
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
            plt.scatter(
                features_array[idx, 0],
                features_array[idx, 1],
                s=5,
                alpha=0.5,
                label=label
            )
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
        results_dir = f"results_{self.name}"
        os.makedirs(results_dir, exist_ok=True)
        plot_filename = os.path.join(results_dir, "diagnostic_plots.png")
        plt.savefig(plot_filename, dpi=300)

        plt.show()

    def plot_templates(self, plotted_var: int, time_span: Optional[tuple] = None):
        """
        Plot trajectories for the template initial conditions.

        Args:
            plotted_var (int): Index of the variable to plot
            time_span (tuple, optional): Time range to plot (t_start, t_end)
        """
        if self.cluster_classifier.initial_conditions is None:
            raise ValueError("No template solutions available.")

        # Get trajectories for template initial conditions
        t, y = self.solver.integrate(
            self.ode_system, self.cluster_classifier.initial_conditions)

        plt.figure(figsize=(8, 6))

        # Filter time if specified
        if time_span is not None:
            idx = (t >= time_span[0]) & (t <= time_span[1])
        else:
            idx = slice(None)

        # Plot each trajectory
        # Use permute instead of transpose for 3D tensors
        for i, (label, traj) in enumerate(zip(self.cluster_classifier.labels, y.permute(1, 0, 2))):
            plt.plot(t[idx], traj[idx, plotted_var], label=f'{label}')

        plt.xlabel('Time')
        plt.ylabel(f'State {plotted_var}')
        plt.title('Template Trajectories')
        plt.legend()
        plt.grid(True)

        # Save plot
        results_dir = f"results_{self.name}"
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(
            results_dir, "template_trajectories.png"), dpi=300)

        plt.show()

    def save(self, filename: str):
        """
        Save the basin stability results to a JSON file.
        Handles numpy arrays and Solution objects by converting them to standard Python types.

        :param filename: The file path where results will be saved.
        """
        if self.bs_vals is None:
            raise ValueError(
                "No results to save. Please run estimate_bs() first.")

        # Create results directory if it does not exist
        results_dir = f"results_{self.name}"
        os.makedirs(results_dir, exist_ok=True)
        real_filename = os.path.join(results_dir, filename)

        results = {
            "assignments": [sol.label for sol in self.solution],
            "bs_vals": self.bs_vals,
            "Y0": self.Y0,
            "features_array": np.vstack([sol.features for sol in self.solution]),
            "solutions": self.solution,
            "N": self.N,
            "steady_state_time": self.feature_extractor.time_steady,
        }

        # Ensure the filename ends with .json
        if not real_filename.endswith('.json'):
            real_filename += '.json'

        with open(real_filename, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)

        print(f"Results saved to {real_filename}")


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, Solution):
            return {
                "initial_condition": obj.initial_condition.tolist(),
                "time": obj.time.tolist(),
                "trajectory": obj.trajectory.tolist(),
                "features": obj.features.tolist() if obj.features is not None else None,
                "label": obj.label
            }
        return super(NumpyEncoder, self).default(obj)
