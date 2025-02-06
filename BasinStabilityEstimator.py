import numpy as np
import matplotlib.pyplot as plt
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict

# It is assumed that these functions/classes are defined elsewhere:
#   - integrate_sample(i, Y0, ode_system, solver, steady_state_time) -> feature vector (np.ndarray)
#   - cluster_assign(features_array, supervised: bool, templates: Optional[np.ndarray]) -> np.ndarray of assignments
#   - features_pendulum (used inside integrate_sample)
#   - ODE system class, Sampler, Solver, etc.
#   - OHE dictionary for default templates, e.g., OHE = {"FP": [1, 0], "LC": [0, 1]}

class BasinStabilityEstimator:
    """
    BasinStabilityEstimator (BSE): The core functionality of this basin stability library.

    This class configures the analysis with objects such as the ODE system, sampler, and solver.
    It provides:
        - estimate_bs() to perform the analysis (like model.fit() in ML)
        - plots() to show diagnostic figures (e.g., basins of attraction, feature space, sampling points)
        - save() to persist the results to a file

    Attributes:
        bs_vals (Optional[Dict[int, float]]): Basin stability values (fraction of samples per class).
        num_pts (int): The number of samples (initial conditions) used.
        assignments (np.ndarray): Class assignment for each sample.
        Y0 (np.ndarray): Array of initial conditions.
        features_array (np.ndarray): Array of features extracted from each integrated trajectory.
    """
    def __init__(
        self,
        N: int,
        steady_state_time: float,
        ode_system,   # expected to be an instance of ODESystem (or similar)
        sampler,      # expected to be an instance of Sampler
        solver,       # expected to be an instance of Solver
        supervised: bool = True,
        templates: Optional[np.ndarray] = None,
    ):
        """
        Initialize the BasinStabilityEstimator.

        :param N: Number of initial conditions (samples) to generate.
        :param steady_state_time: Time after which steady-state features are extracted.
        :param ode_system: The ODE system model.
        :param sampler: The Sampler object to generate initial conditions.
        :param solver: The Solver object to integrate the ODE system.
        :param supervised: Whether to perform supervised clustering/classification.
        :param templates: Templates for supervised clustering (if any).
        """
        self.N = N
        self.steady_state_time = steady_state_time
        self.ode_system = ode_system
        self.sampler = sampler
        self.solver = solver
        self.supervised = supervised
        self.templates = templates

        # Attributes to be populated after estimation
        self.bs_vals: Optional[Dict[int, float]] = None
        self.num_pts: int = N
        self.assignments = None
        self.Y0 = None
        self.features_array = None

    def estimate_bs(self) -> Dict[int, float]:
        """
        Estimate the basin stability by:
          1. Generating initial conditions using the sampler.
          2. Integrating the ODE system for each sample (in parallel).
          3. Extracting features from each trajectory.
          4. Clustering/classifying the features.
          5. Computing the fraction of samples in each basin.

        The method sets the following attributes:
            - self.Y0
            - self.features_array
            - self.assignments
            - self.bs_vals

        :return: A dictionary of basin stability values per class.
        """
        # Step 1: Generate initial conditions.
        self.Y0 = self.sampler.sample(self.N)

        # Step 2/3: Integrate and extract features.
        with ProcessPoolExecutor() as executor:
            # The arguments are broadcast so that each call of integrate_sample gets the same
            # ode_system, solver, and steady_state_time. The index i is passed uniquely.
            all_features = list(executor.map(
                integrate_sample,
                range(self.N),
                [self.Y0] * self.N,
                [self.ode_system] * self.N,
                [self.solver] * self.N,
                [self.steady_state_time] * self.N
            ))
        self.features_array = np.vstack(all_features)  # shape => (N, number_of_features)

        # Step 4: If supervised and no templates provided, try to set default templates.
        if self.supervised and self.templates is None:
            # Make sure that a default OHE dictionary is defined somewhere in your code base.
            try:
                # Example default: 2 classes, Fixed Point and Limit Cycle.
                global OHE
                self.templates = np.array([OHE["FP"], OHE["LC"]], dtype=np.float64)
            except NameError:
                raise ValueError("Templates not provided and default OHE is not defined. Please supply templates.")

        # Step 4: Classify/cluster the feature space.
        self.assignments = cluster_assign(self.features_array, supervised=self.supervised, templates=self.templates)

        # Step 5: Compute basin stability as the fraction of samples for each class.
        unique_labels, counts = np.unique(self.assignments, return_counts=True)
        fractions = counts / float(self.N)
        self.bs_vals = dict(zip(unique_labels.tolist(), fractions.tolist()))

        return self.bs_vals

    def plots(self):
        """
        Generate diagnostic plots:
            1. A bar plot of basin stability values.
            2. A scatter plot of initial conditions (state space).
            3. A scatter plot of the feature space with classifier results.
            4. A placeholder for a future plot.
        """
        if self.bs_vals is None or self.assignments is None or self.Y0 is None or self.features_array is None:
            raise ValueError("No results available. Please run estimate_bs() before plotting.")

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
        unique_labels = np.unique(self.assignments)
        for label in unique_labels:
            idx = self.assignments == label
            plt.scatter(
                self.Y0[idx, 0], self.Y0[idx, 1],
                s=5, alpha=0.5, label=f"Class {label}"
            )
        plt.title("Initial Conditions in State Space")
        plt.xlabel("theta")
        plt.ylabel("theta_dot")
        plt.legend()

        # 3) Feature space scatter plot with classifier results.
        plt.subplot(2, 2, 3)
        for label in unique_labels:
            idx = self.assignments == label
            class_name = "Fixed Point" if label == 0 else "Limit Cycle"
            plt.scatter(
                self.features_array[idx, 0], self.features_array[idx, 1],
                s=5, alpha=0.5, label=class_name
            )
        plt.title("Feature Space with Classifier Results")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()

        # 4) Empty plot for future use.
        plt.subplot(2, 2, 4)
        plt.title("Future Plot")

        plt.tight_layout()
        plt.show()

    def save(self, filename: str):
        """
        Save the basin stability results to a file.

        Currently, pickle is used for serialization.
        (Future work might involve supporting additional file formats.)

        :param filename: The file path where results will be saved.
        """
        if self.bs_vals is None:
            raise ValueError("No results to save. Please run estimate_bs() first.")

        results = {
            "assignments": self.assignments,
            "bs_vals": self.bs_vals,
            "Y0": self.Y0,
            "features_array": self.features_array,
            "N": self.N,
            "steady_state_time": self.steady_state_time,
        }
        with open(filename, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {filename}")

# =============================================================================
# Example usage (ensure that the necessary helper functions and classes are imported):
#
# N = 100
# steady_state_time = 950.0
#
# # Instantiate your ODE system, sampler, and solver:
# ode_system = PendulumODE(params)  # for example
# min_limits = [-np.pi, -2.0]
# max_limits = [np.pi, 2.0]
# sampler = RandomSampler(min_limits, max_limits)
# solver = SciPySolver(time_span=(0, 1000), method="RK45", rtol=1e-8)
#
# # For supervised clustering, you might need default templates:
# OHE = {"FP": [1, 0], "LC": [0, 1]}
# templates = np.array([OHE["FP"], OHE["LC"]], dtype=np.float64)
#
# bse = BasinStabilityEstimator(
#     N=N,
#     steady_state_time=steady_state_time,
#     ode_system=ode_system,
#     sampler=sampler,
#     solver=solver,
#     supervised=True,
#     templates=templates  # or None to trigger default templates (if OHE is defined)
# )
#
# basin_stability = bse.estimate_bs()
# print("Basin Stability:", basin_stability)
# bse.plots()
# bse.save("basin_stability_results.pkl")
# =============================================================================
