import numpy as np
import matplotlib.pyplot as plt
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict, Union

from ClusterClassifier import ClusterClassifier
from FeatureExtractor import FeatureExtractor
from ODESystem import ODESystem

from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from ODESystem import ODESystem
from Sampler import Sampler
from Solution import Solution
from Solver import Solver
import json
from json import JSONEncoder


class BasinStabilityEstimator:
    """
    BasinStabilityEstimator (BSE): Core class for basin stability analysis.
    
    This class configures the analysis with an ODE system, sampler, and solver,
    and it provides methods to estimate the basin stability (estimate_bs),
    generate diagnostic plots (plots), and save results to file (save).
    
    Attributes:
        bs_vals (Optional[Dict[int, float]]): Basin stability values (fraction of samples per class).
        num_pts (int): Number of samples (initial conditions).
        assignments (np.ndarray): Cluster/class assignments for each sample.
        Y0 (np.ndarray): Array of initial conditions.
        features_array (np.ndarray): Matrix of features extracted from each integrated trajectory.
        solutions (List[Solution]): List of Solution instances (one per initial condition).
    """
    def __init__(
        self,
        N: int,
        ode_system: ODESystem,
        sampler: Sampler,
        solver: Solver,
        feature_extractor: FeatureExtractor,
        cluster_classifier: ClusterClassifier,
    ):
        """
        Initialize the BasinStabilityEstimator.
        
        :param N: Number of initial conditions (samples) to generate.
        :param steady_state_time: Time after which steady-state features are extracted.
        :param ode_system: The ODE system model.
        :param sampler: The Sampler object to generate initial conditions.
        :param solver: The Solver object to integrate the ODE system.
        :param cluster_classifier: The ClusterClassifier object to assign labels.
        """
        self.N = N
        self.ode_system = ode_system
        self.sampler = sampler
        self.solver = solver
        self.feature_extractor = feature_extractor
        self.cluster_classifier = cluster_classifier



        # Attributes to be populated during estimation
        self.bs_vals: Optional[Dict[int, float]] = None
        self.Y0 = None
        self.solutions = None  # List of Solution instances

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
            - self.solutions
            - self.bs_vals
        
        :return: A dictionary of basin stability values per class.
        """
        # Step 1: Generate initial conditions.
        self.Y0 = self.sampler.sample(self.N)
        
        # Step 2/3: Integrate and extract features (each integration returns a Solution).
        with ProcessPoolExecutor() as executor:
            solutions = list(executor.map(
                self._integrate_sample,
                range(self.N),
                self.Y0,
                [self.ode_system] * self.N,
                [self.solver] * self.N,
            ))
        self.solutions = solutions
        
        # Build the features array from the Solution instances.
        features_array = np.vstack([sol.features for sol in solutions])
        
        # Step 4: Perform clustering/classification.
        self.cluster_classifier.fit(
            solver=self.solver,
            ode_system=self.ode_system,
            feature_extractor=self.feature_extractor)
        
        assignments = self.cluster_classifier.get_labels(features_array)
        
        # Update each Solution instance with its label.
        for sol, label in zip(self.solutions, assignments):
            sol.assign_label(label)
        
        # Step 5: Compute basin stability as the fraction of samples per class.
        unique_labels, counts = np.unique(assignments, return_counts=True)
        fractions = counts / float(self.N)
        self.bs_vals = dict(zip(unique_labels.tolist(), fractions.tolist()))
        
        return self.bs_vals

    def _integrate_sample(self, i, y0, ode_system, solver) -> Solution:
        """
        Integrate a single sample and return a Solution instance.
        
        :param i: Index of the sample.
        :param Y0: Array of initial conditions.
        :param ode_system: The ODE system model.
        :param solver: The Solver instance for integration.
        :return: A Solution instance.
        """
        # y0 = Y0[i, :]
        print(f"Integrating sample {i+1}/{self.N} with initial condition {y0}")
        
        # Define the ODE system lambda function
        ode_lambda = lambda t, y: ode_system.ode(t, y)
        
        # Perform integration
        t, y = solver.integrate(ode_lambda, y0)
        
        # Create and return a Solution instance
        solution = Solution(
            initial_condition=y0,
            time=t,
            y=y,
        )

        # Extract features (for example, using a feature extractor function)
        features = self.feature_extractor.extract_features(solution)

        solution.set_features(features)

        return solution
        
        
    
    def plots(self):
        """
        Generate diagnostic plots using the data stored in self.solutions:
            1. A bar plot of basin stability values.
            2. A scatter plot of initial conditions (state space).
            3. A scatter plot of the feature space with classifier results.
            4. A placeholder plot for future use.
        """
        if self.solutions is None:
            raise ValueError("No solutions available. Please run estimate_bs() before plotting.")

        # Extract data from each Solution instance.
        initial_conditions = np.array([sol.initial_condition for sol in self.solutions])
        features_array = np.array([sol.features for sol in self.solutions])
        assignments = np.array([sol.label for sol in self.solutions])

        plt.figure(figsize=(10, 6))

        # 1) Bar plot for basin stability values.
        plt.subplot(2, 2, 1)
        labels, values = zip(*self.bs_vals.items())
        plt.bar(labels, values, color=["#1f77b4", "#ff7f0e"])
        plt.xticks(labels)
        plt.ylabel("Fraction of samples")
        plt.title("Basin Stability")

        # 2) State space scatter plot: class-labeled initial conditions.
        plt.subplot(2, 2, 2)
        unique_labels = np.unique(assignments)
        for label in unique_labels:
            idx = assignments == label
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
            idx = assignments == label
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
        plt.show()

    def plot_templates(self, plotted_var: int, time_span: tuple, y_lims: Optional[Union[tuple, list]] = None):
        """
        Plot the time integrations of template solutions.
        Shows trajectory of the first state variable for each template solution.

        Args:
            plotted_var (int): Index of the variable to plot
            time_span (tuple): Time range to plot (t_start, t_end)
            y_lims (tuple or list, optional): Y-axis limits. If tuple, applies to all plots.
                                            If list of tuples, applies individually to each plot.
        """
        if self.cluster_classifier.initial_conditions is None:
            raise ValueError("No template solutions available.")
        
        if time_span is not None:
            assert time_span[0] >= 0, "Time span must be non-negative."
            assert time_span[1] > time_span[0], "Time span must be positive."

        n_solutions = len(self.cluster_classifier.initial_conditions)

        # Validate y_lims if provided
        if y_lims is not None:
            if isinstance(y_lims, tuple):
                y_lims = [y_lims] * n_solutions
            else:
                assert len(y_lims) == n_solutions, f"Length of y_lims ({len(y_lims)}) must match number of solutions ({n_solutions})"
                assert all(isinstance(lim, tuple) for lim in y_lims), "All elements in y_lims must be tuples"
        
        plt.figure(figsize=(4, 1.5*n_solutions))

        # Generate template trajectories and plot
        for i, (ic, label) in enumerate(zip(
            self.cluster_classifier.initial_conditions, 
            self.cluster_classifier.labels
        ), 1):
            # Integrate template
            ode_lambda = lambda t, y: self.ode_system.ode(t, y)
            t, y = self.solver.integrate(ode_lambda, np.array(ic))
            
            if time_span is not None:
                max_ts = t[len(t) - 1]
                assert time_span[1] <= max_ts, f"Time span exceeds maximum time {max_ts}"

            # Find the indices corresponding to the time span
            idx = (t >= time_span[0]) & (t <= time_span[1])

            # Plot only first state variable
            ax = plt.subplot(n_solutions, 1, i)
            ax.plot(t[idx], y[idx, plotted_var])  # y[:, 0] is the first state
            ax.set_xlabel('time')
            ax.set_ylabel('state 1')
            ax.set_title(f'Class {i}, Label: {label}')

            # Apply y-axis limits if specified
            if y_lims is not None:
                ax.set_ylim(y_lims[i-1])

        plt.tight_layout()
        plt.show()
    
    def save(self, filename: str):
        """
        Save the basin stability results to a JSON file.
        Handles numpy arrays and Solution objects by converting them to standard Python types.
        
        :param filename: The file path where results will be saved.
        """
        if self.bs_vals is None:
            raise ValueError("No results to save. Please run estimate_bs() first.")

        results = {
            "assignments": [sol.label for sol in self.solutions],
            "bs_vals": self.bs_vals,
            "Y0": self.Y0,
            "features_array": np.vstack([sol.features for sol in self.solutions]),
            "solutions": self.solutions,
            "N": self.N,
            "steady_state_time": self.feature_extractor.time_steady,
        }

        # Ensure the filename ends with .json
        if not filename.endswith('.json'):
            filename += '.json'

        with open(filename, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)
        
        print(f"Results saved to {filename}")

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
