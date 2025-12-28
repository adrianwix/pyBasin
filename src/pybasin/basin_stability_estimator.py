import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from sklearn.base import BaseEstimator

from pybasin.feature_extractors import TorchFeatureExtractor
from pybasin.feature_extractors.default_feature_selector import DefaultFeatureSelector
from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.feature_extractors.torch_feature_calculators import DEFAULT_TORCH_FC_PARAMETERS
from pybasin.jax_ode_system import JaxODESystem
from pybasin.predictors.base import ClassifierPredictor, LabelPredictor
from pybasin.predictors.hdbscan_clusterer import HDBSCANClusterer
from pybasin.predictors.unboundedness_clusterer import UnboundednessClusterer
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.sampler import Sampler
from pybasin.solution import Solution
from pybasin.solver import TorchDiffEqSolver
from pybasin.solvers.jax_solver import JaxSolver
from pybasin.utils import (
    NumpyEncoder,
    extract_amplitudes,
    generate_filename,
    get_filtered_feature_names,
    resolve_folder,
)

# Sentinel value to distinguish "not specified" from "None"
_USE_DEFAULT = object()


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
        ode_system: ODESystemProtocol,
        sampler: Sampler,
        solver: SolverProtocol | None = None,
        feature_extractor: FeatureExtractor | None = None,
        cluster_classifier: LabelPredictor | None = None,
        feature_selector: BaseEstimator | None = _USE_DEFAULT,  # type: ignore[assignment]
        detect_unbounded: bool = True,
        save_to: str | None = None,
    ):
        """
        Initialize the BasinStabilityEstimator.

        :param n: Number of initial conditions (samples) to generate.
        :param ode_system: The ODE system model (ODESystem or JaxODESystem).
        :param sampler: The Sampler object to generate initial conditions.
        :param solver: The Solver object to integrate the ODE system (Solver or JaxSolver).
                      If None, automatically instantiates JaxSolver for JaxODESystem or
                      TorchDiffEqSolver for ODESystem with time_span=(0, 1000), n_steps=1000,
                      and device from sampler.
        :param feature_extractor: The FeatureExtractor object to extract features from trajectories.
                                 If None, defaults to TorchFeatureExtractor with minimal+dynamical features.
        :param cluster_classifier: The LabelPredictor object to assign labels. If None, defaults
                                  to HDBSCANClusterer with auto_tune=True and assign_noise=True.
        :param feature_selector: Feature filtering sklearn transformer with get_support() method.
                                Defaults to DefaultFeatureSelector(). Pass None to disable filtering.
                                Accepts any sklearn transformer (VarianceThreshold, SelectKBest, etc.) or Pipeline.
        :param detect_unbounded: Enable unboundedness detection before feature extraction (default: True).
                                Only activates when solver has event_fn configured (e.g., JaxSolver with event_fn).
                                When enabled, unbounded trajectories are separated and labeled as "unbounded"
                                before feature extraction to prevent imputed Inf values from contaminating features.
        :param save_to: Optional file path to save results.
        """
        self.n = int(n)
        self.ode_system = ode_system
        self.sampler = sampler
        self.save_to = save_to

        if solver is not None:
            self.solver = solver
        elif isinstance(ode_system, JaxODESystem):
            self.solver = JaxSolver(
                time_span=(0, 1000),
                n_steps=1000,
                device=str(sampler.device),
                use_cache=True,
            )
        else:
            self.solver = TorchDiffEqSolver(
                time_span=(0, 1000),
                n_steps=1000,
                device=str(sampler.device),
                use_cache=True,
            )

        # Initialize feature selector
        # Note: _USE_DEFAULT sentinel distinguishes "not specified" from "None" (disabled)
        if feature_selector is _USE_DEFAULT:
            # Default: use feature filtering with default thresholds
            self.feature_selector: BaseEstimator | None = DefaultFeatureSelector()
        else:
            # User explicitly set it (could be None to disable, or a custom selector)
            self.feature_selector = feature_selector

        self._filtered_feature_names: list[str] | None = None

        # Unboundedness detection: enabled only if detect_unbounded=True AND solver is JaxSolver with event_fn
        self.detect_unbounded = (
            detect_unbounded
            and isinstance(self.solver, JaxSolver)
            and self.solver.event_fn is not None
        )

        if feature_extractor is None:
            time_steady = self.solver.time_span[0] + 0.85 * (
                self.solver.time_span[1] - self.solver.time_span[0]
            )
            # Get device string, with fallback to 'cpu'
            device_str = str(getattr(self.solver, "_device_str", "cpu"))
            feature_extractor = TorchFeatureExtractor(
                time_steady=time_steady,
                features=DEFAULT_TORCH_FC_PARAMETERS,
                device=device_str,  # type: ignore[arg-type]
            )

        self.feature_extractor = feature_extractor

        if cluster_classifier is None:
            warnings.warn(
                "No cluster_classifier provided. Using default HDBSCANClusterer with auto_tune=True "
                "and assign_noise=True. For better performance with known attractors, pass a custom "
                "supervised classifier (e.g., KNNClassifier).",
                UserWarning,
                stacklevel=2,
            )
            cluster_classifier = UnboundednessClusterer(
                HDBSCANClusterer(auto_tune=True, assign_noise=True)
            )
        self.cluster_classifier = cluster_classifier

        # Attributes to be populated during estimation
        self.bs_vals: dict[str, float] | None = None
        self.y0: torch.Tensor | None = None
        self.solution: Solution | None = None

    def _detect_unbounded_trajectories(self, y: torch.Tensor) -> torch.Tensor:
        """Detect unbounded trajectories based on Inf values.

        When JAX Diffrax integration stops due to an event, remaining timesteps are filled with Inf.

        :param y: Trajectory tensor of shape (N, B, S) where N=timesteps, B=batch, S=states.
        :return: Boolean tensor of shape (B,) indicating unbounded trajectories.
        """
        return torch.isinf(y).any(dim=(0, 2))

    def _get_feature_names(self) -> list[str]:
        """Get feature names from extractor.

        :return: List of feature names.
        """
        return self.feature_extractor.feature_names

    def _apply_feature_filtering(
        self, features: torch.Tensor, feature_names: list[str]
    ) -> tuple[torch.Tensor, list[str]]:
        """Apply feature filtering using the configured selector.

        :param features: Feature tensor of shape (n_samples, n_features).
        :param feature_names: List of feature names.
        :return: Tuple of (filtered features tensor, filtered feature names).
        :raises ValueError: If filtering removes all features.
        """
        if self.feature_selector is None:
            return features, feature_names

        # Convert to numpy for sklearn
        features_np = features.detach().cpu().numpy()

        # Apply filtering
        features_filtered_np = cast(
            np.ndarray[Any, np.dtype[np.floating[Any]]],
            self.feature_selector.fit_transform(features_np),  # type: ignore[union-attr]
        )

        # Check if any features remain
        if int(features_filtered_np.shape[1]) == 0:
            raise ValueError(
                f"Feature filtering removed all {features_np.shape[1]} features. "
                "Consider lowering variance_threshold or correlation_threshold."
            )

        # Get filtered feature names using utility function
        filtered_names = get_filtered_feature_names(self.feature_selector, feature_names)

        # Convert back to tensor
        features_filtered = torch.from_numpy(features_filtered_np).to(  # type: ignore[arg-type]
            dtype=features.dtype, device=features.device
        )

        # Print filtering stats
        n_original: int = int(features_np.shape[1])
        n_filtered: int = int(features_filtered_np.shape[1])
        reduction_pct: float = float((1 - n_filtered / n_original) * 100)
        print(
            f"  Feature Filtering: {n_original} → {n_filtered} features "
            f"({reduction_pct:.1f}% reduction)"
        )

        return features_filtered, filtered_names

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

        :param parallel_integration: If True and using ClassifierPredictor, run main integration
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
        t2_start = time.perf_counter()  # Track total integration time
        t2a_elapsed = 0.0  # Template integration time
        t2b_elapsed = 0.0  # Main integration time

        if parallel_integration and isinstance(self.cluster_classifier, ClassifierPredictor):
            print("  Mode: PARALLEL (integration only)")
            print("  • Main integration (sampled ICs)")
            print("  • Template integration (classifier ICs)")

            # Run ONLY integrations in parallel (not feature extraction)
            # This ensures the scaler is fitted on main data first for consistent normalization
            with ThreadPoolExecutor(max_workers=2) as executor:
                main_future = executor.submit(self.solver.integrate, self.ode_system, self.y0)  # type: ignore[arg-type]

                template_future = executor.submit(
                    self.cluster_classifier.integrate_templates,  # type: ignore[arg-type,misc]
                    self.solver,
                    self.ode_system,
                )

                # Wait for both to complete
                t, y = main_future.result()
                template_future.result()  # Just wait for completion

            t2_elapsed = time.perf_counter() - t2_start
            # In parallel mode, we can't separate the times accurately
            print(f"  Both integrations complete in {t2_elapsed:.4f}s")
            print(f"  Main trajectory shape: {y.shape}")
        else:
            # Sequential execution (original behavior)
            if isinstance(self.cluster_classifier, ClassifierPredictor):
                print("  Mode: SEQUENTIAL")
                print("  Step 2a: Integrating template initial conditions...")
                t2a_start = time.perf_counter()
                self.cluster_classifier.integrate_templates(  # type: ignore[misc]
                    solver=self.solver,
                    ode_system=self.ode_system,
                )
                t2a_elapsed = time.perf_counter() - t2a_start
                print(f"    Template integration in {t2a_elapsed:.4f}s")

            print("  Step 2b: Integrating sampled initial conditions...")
            t2b_start = time.perf_counter()
            t, y = self.solver.integrate(self.ode_system, self.y0)  # type: ignore[arg-type]
            t2b_elapsed = time.perf_counter() - t2b_start
            print(f"    Main trajectory shape: {y.shape}")
            print(f"    Main integration complete in {t2b_elapsed:.4f}s")

            # Total integration time includes both template and main
            t2_elapsed = time.perf_counter() - t2_start
            print(f"    Total integration time: {t2_elapsed:.4f}s")

        # Step 3: Create Solution object
        print("\nSTEP 3: Creating Solution Object")
        t3 = time.perf_counter()
        self.solution = Solution(initial_condition=self.y0, time=t, y=y)

        # Always compute bifurcation amplitudes
        self.solution.bifurcation_amplitudes = extract_amplitudes(t, y)
        t3_elapsed = time.perf_counter() - t3
        print(f"  Solution object created in {t3_elapsed:.4f}s")

        # Step 3b: Detect and separate unbounded trajectories (if enabled)
        unbounded_mask: torch.Tensor | None = None
        n_unbounded = 0
        total_samples = len(self.y0)
        original_solution: Solution | None = None

        if self.detect_unbounded:
            print("\nSTEP 3b: Unboundedness Detection")
            t3b = time.perf_counter()
            unbounded_mask = self._detect_unbounded_trajectories(y)
            n_unbounded = int(unbounded_mask.sum().item())
            n_bounded = total_samples - n_unbounded
            unbounded_pct = (n_unbounded / total_samples) * 100
            t3b_elapsed = time.perf_counter() - t3b

            print(
                f"  Detected {n_unbounded}/{total_samples} unbounded trajectories ({unbounded_pct:.1f}%) in {t3b_elapsed:.4f}s"
            )

            if n_unbounded == total_samples:
                print(
                    "  All trajectories are unbounded. Skipping feature extraction and classification."
                )
                self.bs_vals = {"unbounded": 1.0}
                labels = np.array(["unbounded"] * total_samples, dtype=object)
                self.solution.set_labels(labels)

                total_elapsed = time.perf_counter() - total_start
                print("\nBASIN STABILITY ESTIMATION COMPLETE")
                print(f"Total time: {total_elapsed:.4f}s")
                return self.bs_vals

            if n_unbounded > 0:
                print(
                    f"  Separating {n_bounded} bounded trajectories for feature extraction and classification"
                )
                bounded_mask = ~unbounded_mask

                # Store original solution for later restoration
                original_solution = self.solution

                y0_bounded = self.y0[bounded_mask]
                y_bounded = y[:, bounded_mask, :]

                self.solution = Solution(initial_condition=y0_bounded, time=t, y=y_bounded)
                self.solution.bifurcation_amplitudes = extract_amplitudes(t, y_bounded)
        else:
            print("\n  Unboundedness detection: DISABLED")

        # Step 4: Feature extraction (main data - fits scaler on large dataset)
        print("\nSTEP 4: Feature Extraction")
        t4 = time.perf_counter()
        features = self.feature_extractor.extract_features(self.solution)

        # Get feature names and store extracted features
        feature_names = self._get_feature_names()
        self.solution.set_extracted_features(features, feature_names)
        t4_elapsed = time.perf_counter() - t4
        print(f"  Extracted features with shape {features.shape} in {t4_elapsed:.4f}s")

        # Step 5: Feature filtering
        print("\nSTEP 5: Feature Filtering")
        t5 = time.perf_counter()
        if self.feature_selector is not None:
            features_filtered, filtered_names = self._apply_feature_filtering(
                features, feature_names
            )
            self.solution.set_features(features_filtered, filtered_names)
            self._filtered_feature_names = filtered_names
            features = features_filtered
        else:
            self.solution.set_features(features, feature_names)
            print("  No feature filtering configured")
        t5_elapsed = time.perf_counter() - t5
        print(f"  Feature filtering complete in {t5_elapsed:.4f}s")

        # Show sample of filtered features (first IC, up to 10 features)
        if self.solution.features is not None and self.solution.features.shape[0] > 0:
            n_features_to_show = min(10, self.solution.features.shape[1])
            if n_features_to_show > 0:
                print(f"\n  Sample of first {n_features_to_show} filtered features (first IC):")
                feature_names_filtered = (
                    self.solution.filtered_feature_names[:n_features_to_show]
                    if self.solution.filtered_feature_names
                    else []
                )
                feature_values: list[float] = (
                    self.solution.features[0, :n_features_to_show].cpu().numpy().tolist()
                )
                for name, value in zip(feature_names_filtered, feature_values, strict=False):
                    print(f"    {name}: {value:.6f}")

        # Step 5b: Fit classifier with template features (using already-fitted scaler)
        if isinstance(self.cluster_classifier, ClassifierPredictor):  # type: ignore[type-arg]
            print("\nSTEP 5b: Fitting Classifier")
            t5b = time.perf_counter()
            self.cluster_classifier.fit_with_features(  # type: ignore[misc]
                self.feature_extractor,
                feature_selector=self.feature_selector,
            )
            t5b_elapsed = time.perf_counter() - t5b
            print(f"  Classifier fitted in {t5b_elapsed:.4f}s")

        # Step 6: Classification
        print("\nSTEP 6: Classification")
        t6 = time.perf_counter()

        # Convert features to numpy for classifier
        t6_pred = time.perf_counter()
        features_np = features.detach().cpu().numpy()
        bounded_labels = self.cluster_classifier.predict_labels(features_np)  # type: ignore[misc]
        t6_pred_elapsed = time.perf_counter() - t6_pred

        # Reconstruct full label array if unbounded trajectories were separated
        if self.detect_unbounded and unbounded_mask is not None and n_unbounded > 0:
            labels = np.empty(total_samples, dtype=object)
            labels[unbounded_mask.cpu().numpy()] = "unbounded"
            labels[~unbounded_mask.cpu().numpy()] = bounded_labels
            print(f"  Classified {len(bounded_labels)} bounded trajectories")
            print(f"  Reconstructed full label array with {n_unbounded} unbounded labels")

            # Restore original solution with full trajectories
            if original_solution is not None:
                self.solution = original_solution
        else:
            labels = bounded_labels

        self.solution.set_labels(labels)
        t6_elapsed = time.perf_counter() - t6
        print(f"  Classification complete in {t6_elapsed:.4f}s")
        print(f"  Prediction time: {t6_pred_elapsed:.4f}s")

        # Step 7: Computing Basin Stability
        print("\nSTEP 7: Computing Basin Stability")
        t7 = time.perf_counter()

        # Convert all labels to strings to ensure consistent types (bounded labels may be int or str)
        labels_str = np.array([str(label) for label in labels], dtype=object)
        unique_labels, counts = np.unique(labels_str, return_counts=True)

        self.bs_vals = {str(label): 0.0 for label in unique_labels}

        # Use the actual number of samples generated, not the requested n
        # This is important because GridSampler may generate more points than requested
        actual_n = len(labels)
        fractions = counts / float(actual_n)

        for label, fraction in zip(unique_labels, fractions, strict=True):
            basin_stability_fraction = float(fraction)
            self.bs_vals[str(label)] = basin_stability_fraction
            print(f"  {label}: {basin_stability_fraction * 100:.2f}%")

        t7_elapsed = time.perf_counter() - t7
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
        if t2a_elapsed > 0:
            print(
                f"     - Template:         {t2a_elapsed:8.4f}s  ({t2a_elapsed / total_elapsed * 100:5.1f}%)"
            )
        if t2b_elapsed > 0:
            print(
                f"     - Main:             {t2b_elapsed:8.4f}s  ({t2b_elapsed / total_elapsed * 100:5.1f}%)"
            )
        print(
            f"  3. Solution/Amps:      {t3_elapsed:8.4f}s  ({t3_elapsed / total_elapsed * 100:5.1f}%)"
        )
        print(
            f"  4. Features:           {t4_elapsed:8.4f}s  ({t4_elapsed / total_elapsed * 100:5.1f}%)"
        )
        print(
            f"  5. Filtering:          {t5_elapsed:8.4f}s  ({t5_elapsed / total_elapsed * 100:5.1f}%)"
        )
        print(
            f"  6. Classification:     {t6_elapsed:8.4f}s  ({t6_elapsed / total_elapsed * 100:5.1f}%)"
        )
        print(
            f"  7. BS Computation:     {t7_elapsed:8.4f}s  ({t7_elapsed / total_elapsed * 100:5.1f}%)"
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

        # Feature selection information
        feature_selection_info: dict[str, Any] = {
            "enabled": self.feature_selector is not None,
        }

        if self.feature_selector is not None:
            feature_selection_info["selector_type"] = type(self.feature_selector).__name__

            # Add feature count information
            if self.solution and self.solution.extracted_features is not None:
                n_extracted = self.solution.extracted_features.shape[1]
                n_filtered = (
                    self.solution.features.shape[1] if self.solution.features is not None else 0
                )
                feature_selection_info["n_features_extracted"] = n_extracted
                feature_selection_info["n_features_filtered"] = n_filtered
                feature_selection_info["reduction_ratio"] = (
                    (1 - n_filtered / n_extracted) if n_extracted > 0 else 0.0
                )

                if self._filtered_feature_names:
                    feature_selection_info["filtered_feature_names"] = self._filtered_feature_names
        else:
            feature_selection_info["selector_type"] = "disabled"

        results: dict[str, Any] = {
            "basin_of_attractions": self.bs_vals,
            "region_of_interest": region_of_interest,
            "sampling_points": self.n,
            "sampling_method": self.sampler.__class__.__name__,
            "solver": self.solver.__class__.__name__,
            "cluster_classifier": self.cluster_classifier.__class__.__name__,
            "feature_selection": feature_selection_info,
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
        y0_list: list[Any] = self.y0.detach().cpu().numpy().tolist()
        amplitudes_list: list[Any] = (
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

        df.to_excel(full_path, index=False)  # type: ignore[call-overload]
