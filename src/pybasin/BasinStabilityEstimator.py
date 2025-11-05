
import json
from typing import Dict, Optional
from typing import Optional, Dict
import os
import numpy as np
import pandas as pd


from pybasin.Solver import Solver
from pybasin.Solution import Solution
from pybasin.Sampler import Sampler
from pybasin.ODESystem import ODESystem
from pybasin.FeatureExtractor import FeatureExtractor
from pybasin.ClusterClassifier import ClusterClassifier, SupervisedClassifier
from pybasin.utils import NumpyEncoder, extract_amplitudes, generate_filename, resolve_folder


class BasinStabilityEstimator:
    """
    BasinStabilityEstimator (BSE): Core class for basin stability analysis.

    This class configures the analysis with an ODE system, sampler, and solver,
    and it provides methods to estimate the basin stability (estimate_bs), and save results to file (save).

    Attributes:
        bs_vals (Optional[Dict[int, float]]): Basin stability values (fraction of samples per class).
        Y0 (np.ndarray): Array of initial conditions.
        solution (Solution): Solution instance.
    """
    solution: Solution
    bs_vals: Optional[Dict[int, float]]

    def __init__(
        self,
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

        :param N: Number of initial conditions (samples) to generate.
        :param steady_state_time: Time after which steady-state features are extracted.
        :param ode_system: The ODE system model.
        :param sampler: The Sampler object to generate initial conditions.
        :param solver: The Solver object to integrate the ODE system.
        :param cluster_classifier: The ClusterClassifier object to assign labels.
        :param save_to: Optional file path to save results.
        """
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
        self.amplitude_extractor = None

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

        if self.amplitude_extractor is None:
            self.solution.bifurcation_amplitudes = extract_amplitudes(t, y)

        print("\n4. Extracting features...")
        features = self.feature_extractor.extract_features(self.solution)
        self.solution.set_features(features)
        print(f"   Features shape: {features.shape}")

        print("\n5. Performing classification...")
        if isinstance(self.cluster_classifier, SupervisedClassifier):
            print("   Fitting classifier with template data...")
            self.cluster_classifier.fit(
                solver=self.solver,
                ode_system=self.ode_system,
                feature_extractor=self.feature_extractor)

        labels = self.cluster_classifier.predict_labels(features)
        self.solution.set_labels(labels)
        print("   Classification complete")

        print("\n6. Computing basin stability values...")
        unique_labels, counts = np.unique(labels, return_counts=True)

        self.bs_vals = {
            str(label): 0.0 for label in unique_labels}

        fractions = counts / float(self.N)

        for label, fraction in zip(unique_labels, fractions):
            self.bs_vals[str(label)] = fraction
            print(f"   {label}: {fraction:.3f}")

        print("\nBasin Stability Estimation Complete!")
        return self.bs_vals

    def save(self):
        """
        Save the basin stability results to a JSON file.
        Handles numpy arrays and Solution objects by converting them to standard Python types.

        :param filename: The file path where results will be saved.
        """
        if self.bs_vals is None:
            raise ValueError(
                "No results to save. Please run estimate_bs() first.")

        if self.save_to is None:
            raise ValueError(
                "save_to is not defined.")

        full_folder = resolve_folder(self.save_to)
        file_name = generate_filename('basin_stability_results', 'json')
        full_path = os.path.join(full_folder, file_name)

        def format_ode_system(ode_str: str) -> list:
            lines = ode_str.strip().split('\n')
            formatted_lines = [' '.join(line.split()) for line in lines]
            return formatted_lines

        region_of_interest = " X ".join(
            [f"[{min_val}, {max_val}]" for min_val, max_val in zip(
                self.sampler.min_limits, self.sampler.max_limits)]
        )

        results = {
            "basin_of_attractions": self.bs_vals,
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

    def save_to_excel(self):
        if self.bs_vals is None:
            raise ValueError(
                "No results to save. Please run estimate_bs() first.")

        if self.save_to is None:
            raise ValueError(
                "save_to is not defined.")

        full_folder = resolve_folder(self.save_to)
        file_name = generate_filename('basin_stability_results', 'xlsx')
        full_path = os.path.join(full_folder, file_name)

        df = pd.DataFrame({
            'Grid Sample': [(x, y) for x, y in self.Y0.tolist()],
            'Labels': self.solution.labels,
            'Bifurcation Amplitudes': [
                (theta, theta_dot)
                for theta, theta_dot in self.solution.bifurcation_amplitudes.tolist()]
        })

        df.to_excel(full_path)
