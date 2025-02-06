import numpy as np
import matplotlib.pyplot as plt
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict

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
from Solution import Solution

# OHE = {
#     "FP": np.array([1.0, 0.0], dtype=np.float64),
#     "LC": np.array([0.0, 1.0], dtype=np.float64)
# }
# OHE = {"FP": [1, 0], "LC": [0, 1]}

# def _features_pendulum(
#     t: NDArray[np.float64],
#     y: NDArray[np.float64],
#     steady_state_time: float = 950.0
# ) -> NDArray[np.float64]:
#     """
#     Replicates the MATLAB 'features_pendulum' function in Python:
#       1) Identify time indices for t > steady_state_time (steady-state).
#       2) Compute Delta = |max(theta_dot) - mean(theta_dot)|.
#       3) If Delta < 0.01 => [1,0] (FP), else => [0,1] (LC).

#     Parameters
#     ----------
#     t : NDArray[np.float64]
#         Time values from the integration.
#     y : NDArray[np.float64], shape (len(t), 2)
#         The states at each time in t.  y[:,0] = theta, y[:,1] = theta_dot
#     steady_state_time : float
#         Time after which we consider the system to be near steady-state.

#     Returns
#     -------
#     X : NDArray[np.float64], shape (2, 1)
#         A one-hot vector, [1,0]^T for FP or [0,1]^T for LC.
#     """
#     # Indices where t > steady_state_time
#     idx_steady = np.where(t > steady_state_time)[0]
#     if len(idx_steady) == 0:
#         print("Warning: No steady state found.")
#         # If we never get beyond steady_state_time, default to [1,0]
#         return np.array(OHE["FP"], dtype=np.float64)

#     print(f"Steady state found at t={t[idx_steady[0]]}")
#     idx_start = idx_steady[0]

#     # Use the second state (theta_dot) for the portion after steady_state_time
#     portion = y[idx_start:, 1]
    
#     delta = np.abs(np.max(portion) - np.mean(portion))
#     print(f"Delta = {delta}")

#     if delta < 0.01:
#         print("Fixed Point (FP)")
#         # FP (Fixed Point)
#         return np.array(OHE["FP"], dtype=np.float64)
#     else:
#         # LC (Limit Cycle)
#         return np.array(OHE["LC"], dtype=np.float64)


def cluster_assign(
    features: NDArray[np.float64],
    supervised: bool = True,
    templates: Optional[NDArray[np.float64]] = None
) -> NDArray[np.int64]:
    """
    Assign a cluster/label to each feature vector.

    In 'supervised' mode with known templates, we use k-Nearest Neighbors (k=1).
    Otherwise, we do an unsupervised KMeans with 2 clusters.

    Parameters
    ----------
    features : NDArray[np.float64], shape (N, num_features)
        Extracted features for each sample (row).
    supervised : bool
        If True, do supervised classification (requires 'templates').
        If False, do KMeans with 2 clusters.
    templates : Optional[NDArray[np.float64]]
        Labeled samples for training in the supervised scenario. 
        E.g. np.array([[1, 0], [0, 1]]) with implicit labels 0 => FP, 1 => LC.

    Returns
    -------
    assignments : NDArray[np.int64], shape (N,)
        The integer label for each feature row.
    """
    # TODO: This function looks wrong
    if supervised and (templates is not None):
        # Suppose templates is [[1,0], [0,1]] => we map them to y_template=[0,1]
        X_template = templates
        y_template = np.array(["Fixed Point", "Limit Cycle"])

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_template, y_template)
        assignments = knn.predict(features)
        return assignments
    else:
        # Unsupervised KMeans with 2 clusters
        kmeans = KMeans(n_clusters=2, n_init="auto")
        kmeans.fit(features)
        return kmeans.labels_.astype(np.int64)


# It is assumed that these functions/classes are defined elsewhere:
#   - integrate_sample(i, Y0, ode_system, solver, steady_state_time) -> feature vector (np.ndarray)
#   - cluster_assign(features_array, supervised: bool, templates: Optional[np.ndarray]) -> np.ndarray of assignments
#   - features_pendulum (used inside integrate_sample)
#   - ODE system class, Sampler, Solver, etc.
#   - OHE dictionary for default templates, e.g., OHE = {"FP": [1, 0], "LC": [0, 1]}


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
        ode_system,   # Expected to be an instance of an ODE system class.
        sampler,      # Expected to be an instance of Sampler.
        solver,       # Expected to be an instance of Solver.
        feature_extractor: FeatureExtractor,
        # TODO: extract cluster class  
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
        self.ode_system = ode_system
        self.sampler = sampler
        self.solver = solver
        self.feature_extractor = feature_extractor

        self.supervised = supervised
        self.templates = templates

        # Attributes to be populated during estimation
        self.bs_vals: Optional[Dict[int, float]] = None
        self.num_pts: int = N
        self.assignments = None
        self.Y0 = None
        self.features_array = None
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
            - self.features_array
            - self.assignments
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
        self.features_array = np.vstack([sol.features for sol in solutions])
        
        # Step 4: Set default templates if needed (for supervised clustering).
        if self.supervised and self.templates is None:
            try:
                global OHE
                self.templates = np.array([OHE["FP"], OHE["LC"]], dtype=np.float64)
            except NameError:
                raise ValueError("Templates not provided and default OHE is not defined. Please supply templates.")
        
        # Perform clustering/classification.
        self.assignments = cluster_assign(self.features_array, supervised=self.supervised, templates=self.templates)
        
        # Update each Solution instance with its label.
        for sol, label in zip(self.solutions, self.assignments):
            sol.assign_label(label)
        
        # Step 5: Compute basin stability as the fraction of samples per class.
        unique_labels, counts = np.unique(self.assignments, return_counts=True)
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
            "solutions": self.solutions,
            "N": self.N,
            "steady_state_time": self.feature_extractor.time_steady,
        }
        with open(filename, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {filename}")
