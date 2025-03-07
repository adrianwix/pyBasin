import inspect
from json import JSONEncoder
import json
from typing import Dict, Optional
from typing import Optional, Dict
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datetime
import inspect
import uuid

from pybasin.Solver import Solver
from pybasin.Solution import Solution
from pybasin.Sampler import Sampler
from pybasin.ODESystem import ODESystem
from pybasin.FeatureExtractor import FeatureExtractor
from pybasin.ClusterClassifier import ClusterClassifier
from pybasin.utils import NumpyEncoder, generate_filename, resolve_folder

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
        save_to: Optional[str] = None,
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
        :param save_to: Optional file path to save results.
        """
        self.name = name
        self.N = N
        self.ode_system = ode_system
        self.sampler = sampler
        self.solver = solver
        self.feature_extractor = feature_extractor
        self.cluster_classifier = cluster_classifier
        self.save_to = save_to

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
        print("\nStarting Basin Stability Estimation...")

        print("\n1. Generating initial conditions...")
        self.Y0 = self.sampler.sample(self.N)
        print(f"   Generated {self.N} initial conditions")

        print("\n2. Integrating ODE system...")
        t, y = self.solver.integrate(self.ode_system, self.Y0)
        print(f"   Integration complete - trajectory shape: {y.shape}")

        print("\n3. Creating Solution object...")
        self.solution = Solution(
            initial_condition=self.Y0,
            time=t,
            y=y
        )

        print("\n4. Extracting features...")
        features = self.feature_extractor.extract_features(self.solution)
        self.solution.set_features(features)
        print(f"   Features shape: {features.shape}")

        print("\n5. Performing classification...")
        if self.cluster_classifier.type == 'supervised':
            print("   Fitting classifier with template data...")
            self.cluster_classifier.fit(
                solver=self.solver,
                ode_system=self.ode_system,
                feature_extractor=self.feature_extractor)

        labels = self.cluster_classifier.get_labels(features)
        self.solution.set_labels(labels)
        print("   Classification complete")

        print("\n6. Computing basin stability values...")
        self.bs_vals = {str(label): 0.0 for label in labels}
        unique_labels, counts = np.unique(labels, return_counts=True)
        fractions = counts / float(self.N)

        for label, fraction in zip(unique_labels, fractions):
            self.bs_vals[str(label)] = fraction
            print(f"   {label}: {fraction:.3f}")

        print("\nBasin Stability Estimation Complete!")
        return self.bs_vals

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
