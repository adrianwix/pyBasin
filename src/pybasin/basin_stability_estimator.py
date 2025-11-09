import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import torch

from pybasin.cluster_classifier import ClusterClassifier, SupervisedClassifier
from pybasin.feature_extractor import FeatureExtractor
from pybasin.ode_system import ODESystem
from pybasin.sampler import Sampler
from pybasin.solution import Solution
from pybasin.solver import Solver
from pybasin.utils import NumpyEncoder, extract_amplitudes, generate_filename, resolve_folder


class BasinStabilityEstimator:
    """
    BasinStabilityEstimator (BSE): Core class for basin stability analysis.

    This class configures the analysis with an ODE system, sampler, and solver,
    and it provides methods to estimate the basin stability (estimate_bs), and save results to file (save).

    Attributes:
        bs_vals (dict[str, float] | None): Basin stability values (fraction of samples per class).
        y0 (torch.Tensor | None): Array of initial conditions.
        solution (Solution | None): Solution instance containing trajectory and analysis results.
    """

    def __init__(
        self,
        n: int,
        ode_system: ODESystem[Any],
        sampler: Sampler,
        solver: Solver,
        feature_extractor: FeatureExtractor,
        cluster_classifier: ClusterClassifier,
        save_to: str | None = None,
    ):
        """
        Initialize the BasinStabilityEstimator.

        :param n: Number of initial conditions (samples) to generate.
        :param ode_system: The ODE system model.
        :param sampler: The Sampler object to generate initial conditions.
        :param solver: The Solver object to integrate the ODE system.
        :param feature_extractor: The FeatureExtractor object to extract features from trajectories.
        :param cluster_classifier: The ClusterClassifier object to assign labels.
        :param save_to: Optional file path to save results.
        """
        self.n = int(
            n
        )  # Ensure n is always an int (handles numpy.float64 from adaptive parameters)
        self.ode_system = ode_system
        self.sampler = sampler
        self.solver = solver
        self.feature_extractor = feature_extractor
        self.cluster_classifier = cluster_classifier
        self.save_to = save_to

        # Attributes to be populated during estimation
        self.bs_vals: dict[str, float] | None = None
        self.y0: torch.Tensor | None = None
        self.solution: Solution | None = None

    def estimate_bs(self, parallel_integration: bool = True) -> dict[str, float]:
        """
        Estimate basin stability by:
            1. Generating initial conditions using the sampler.
            2. Integrating the ODE system for each sample (in parallel) to produce a Solution.
            3. Extracting features from each Solution.
            4. Clustering/classifying the feature space.
            5. Computing the fraction of samples in each basin.

        This method sets:
            - self.y0
            - self.solution
            - self.bs_vals

        :param parallel_integration: If True and using SupervisedClassifier, run main integration
                                     and template integration in parallel (default: True).
        :return: A dictionary of basin stability values per class.
        """
        print("\nStarting Basin Stability Estimation...")
        total_start = time.perf_counter()

        # Step 1: Sampling
        print("\nSTEP 1: Sampling Initial Conditions")
        t1 = time.perf_counter()
        self.y0 = self.sampler.sample(self.n)
        t1_elapsed = time.perf_counter() - t1
        print(f"  Generated grid with {len(self.y0)} initial conditions in {t1_elapsed:.4f}s")

        # Step 2: Integration (possibly parallel with classifier fitting)
        print("\nSTEP 2: ODE Integration")

        if parallel_integration and isinstance(self.cluster_classifier, SupervisedClassifier):
            print("  Mode: PARALLEL")
            print("  • Main integration (sampled ICs)")
            print("  • Template integration (classifier fitting)")
            t2 = time.perf_counter()

            # Run both integrations in parallel using threads
            # Threads work for GPU operations because PyTorch releases the GIL during CUDA kernels

            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit main integration
                main_future = executor.submit(self.solver.integrate, self.ode_system, self.y0)

                # Submit classifier fitting (which does its own integration)
                classifier_future = executor.submit(
                    self.cluster_classifier.fit,
                    self.solver,
                    self.ode_system,
                    self.feature_extractor,
                )

                # Wait for both to complete
                t, y = main_future.result()
                classifier_future.result()  # Just wait for completion

            t2_elapsed = time.perf_counter() - t2
            print(f"  Both integrations complete in {t2_elapsed:.4f}s")
            print(f"  Main trajectory shape: {y.shape}")
        else:
            # Sequential execution (original behavior)
            if isinstance(self.cluster_classifier, SupervisedClassifier):
                print("  Mode: SEQUENTIAL")
                print("  Step 2a: Fitting classifier with template data...")
                t2a = time.perf_counter()
                self.cluster_classifier.fit(
                    solver=self.solver,
                    ode_system=self.ode_system,
                    feature_extractor=self.feature_extractor,
                )
                t2a_elapsed = time.perf_counter() - t2a
                print(f"    Classifier fitted in {t2a_elapsed:.4f}s")

            print("  Step 2b: Integrating sampled initial conditions...")
            t2 = time.perf_counter()
            t, y = self.solver.integrate(self.ode_system, self.y0)
            t2_elapsed = time.perf_counter() - t2
            print(f"    Main trajectory shape: {y.shape}")
            print(f"    Integration complete in {t2_elapsed:.4f}s")

        # Step 3: Create Solution object
        print("\nSTEP 3: Creating Solution Object")
        t3 = time.perf_counter()
        self.solution = Solution(initial_condition=self.y0, time=t, y=y)

        # Always compute bifurcation amplitudes
        self.solution.bifurcation_amplitudes = extract_amplitudes(t, y)
        t3_elapsed = time.perf_counter() - t3
        print(f"  Solution object created in {t3_elapsed:.4f}s")

        # Step 4: Feature extraction
        print("\nSTEP 4: Feature Extraction")
        t4 = time.perf_counter()
        features = self.feature_extractor.extract_features(self.solution)
        self.solution.set_features(features)
        t4_elapsed = time.perf_counter() - t4
        print(f"  Extracted features with shape {features.shape} in {t4_elapsed:.4f}s")

        # Step 5: Classification
        print("\nSTEP 5: Classification")
        t5 = time.perf_counter()

        # Convert features to numpy for classifier
        t5_pred = time.perf_counter()
        features_np = features.detach().cpu().numpy()
        labels = self.cluster_classifier.predict_labels(features_np)
        self.solution.set_labels(labels)
        t5_pred_elapsed = time.perf_counter() - t5_pred
        t5_elapsed = time.perf_counter() - t5
        print(f"  Classification complete in {t5_elapsed:.4f}s")
        print(f"  Prediction time: {t5_pred_elapsed:.4f}s")

        # Step 6: Compute basin stability
        print("\nSTEP 6: Computing Basin Stability")
        t6 = time.perf_counter()
        unique_labels, counts = np.unique(labels, return_counts=True)

        self.bs_vals = {str(label): 0.0 for label in unique_labels}

        # Use the actual number of samples generated, not the requested n
        # This is important because GridSampler may generate more points than requested
        actual_n = len(labels)
        fractions = counts / float(actual_n)

        for label, fraction in zip(unique_labels, fractions, strict=True):
            self.bs_vals[str(label)] = fraction
            print(f"  {label}: {fraction:.4f}")

        t6_elapsed = time.perf_counter() - t6
        print(f"  Basin stability computed in {t6_elapsed:.4f}s")

        # Summary
        total_elapsed = time.perf_counter() - total_start
        print("\nBASIN STABILITY ESTIMATION COMPLETE")
        print(f"Total time: {total_elapsed:.4f}s")
        print("\nTiming Breakdown:")
        print(
            f"  1. Sampling:           {t1_elapsed:8.4f}s  ({t1_elapsed / total_elapsed * 100:5.1f}%)"
        )
        print(
            f"  2. Integration:        {t2_elapsed:8.4f}s  ({t2_elapsed / total_elapsed * 100:5.1f}%)"
        )
        print(
            f"  3. Solution/Amps:      {t3_elapsed:8.4f}s  ({t3_elapsed / total_elapsed * 100:5.1f}%)"
        )
        print(
            f"  4. Features:           {t4_elapsed:8.4f}s  ({t4_elapsed / total_elapsed * 100:5.1f}%)"
        )
        print(
            f"  5. Classification:     {t5_elapsed:8.4f}s  ({t5_elapsed / total_elapsed * 100:5.1f}%)"
        )
        print(
            f"  6. BS Computation:     {t6_elapsed:8.4f}s  ({t6_elapsed / total_elapsed * 100:5.1f}%)"
        )

        return self.bs_vals

    def save(self) -> None:
        """
        Save the basin stability results to a JSON file.
        Handles numpy arrays and Solution objects by converting them to standard Python types.
        """
        if self.bs_vals is None:
            raise ValueError("No results to save. Please run estimate_bs() first.")

        if self.save_to is None:
            raise ValueError("save_to is not defined.")

        full_folder = resolve_folder(self.save_to)
        file_name = generate_filename("basin_stability_results", "json")
        full_path = os.path.join(full_folder, file_name)

        def format_ode_system(ode_str: str) -> list[str]:
            lines = ode_str.strip().split("\n")
            formatted_lines = [" ".join(line.split()) for line in lines]
            return formatted_lines

        region_of_interest = " X ".join(
            [
                f"[{min_val}, {max_val}]"
                for min_val, max_val in zip(
                    self.sampler.min_limits, self.sampler.max_limits, strict=True
                )
            ]
        )

        results: dict[str, Any] = {
            "basin_of_attractions": self.bs_vals,
            "region_of_interest": region_of_interest,
            "sampling_points": self.n,
            "sampling_method": self.sampler.__class__.__name__,
            "solver": self.solver.__class__.__name__,
            "cluster_classifier": self.cluster_classifier.__class__.__name__,
            "ode_system": format_ode_system(self.ode_system.get_str()),
        }

        with open(full_path, "w") as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)

        print(f"Results saved to {full_path}")

    def save_to_excel(self) -> None:
        """
        Save the basin stability results to an Excel file.
        Includes grid samples, labels, and bifurcation amplitudes.
        """
        if self.bs_vals is None:
            raise ValueError("No results to save. Please run estimate_bs() first.")

        if self.save_to is None:
            raise ValueError("save_to is not defined.")

        if self.y0 is None or self.solution is None:
            raise ValueError("No solution data available. Please run estimate_bs() first.")

        full_folder = resolve_folder(self.save_to)
        file_name = generate_filename("basin_stability_results", "xlsx")
        full_path = os.path.join(full_folder, file_name)

        # Convert tensors to lists for DataFrame
        y0_list = self.y0.detach().cpu().numpy().tolist()
        amplitudes_list = (
            self.solution.bifurcation_amplitudes.detach().cpu().numpy().tolist()
            if self.solution.bifurcation_amplitudes is not None
            else []
        )

        df = pd.DataFrame(
            {
                "Grid Sample": [tuple(ic) for ic in y0_list],
                "Labels": self.solution.labels if self.solution.labels is not None else [],
                "Bifurcation Amplitudes": [tuple(amp) for amp in amplitudes_list],
            }
        )

        df.to_excel(full_path, index=False)
