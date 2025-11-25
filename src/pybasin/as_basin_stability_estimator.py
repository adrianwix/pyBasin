"""Adaptive Study Basin Stability Estimator."""

import gc
import json
import os
from typing import Any, TypedDict

import numpy as np
import torch

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.cluster_classifier import ClusterClassifier
from pybasin.feature_extractor import FeatureExtractor
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.sampler import Sampler
from pybasin.utils import NumpyEncoder, generate_filename, resolve_folder


class AdaptiveStudyParams(TypedDict):
    """Parameters for adaptive parameter study."""

    adaptative_parameter_values: np.ndarray
    adaptative_parameter_name: str


class ASBasinStabilityEstimator:
    """
    Adaptive Study Basin Stability Estimator.
    """

    def __init__(
        self,
        n: int,
        ode_system: ODESystemProtocol,
        sampler: Sampler,
        solver: SolverProtocol,
        feature_extractor: FeatureExtractor,
        cluster_classifier: ClusterClassifier,
        as_params: AdaptiveStudyParams,
        save_to: str | None = "results",
    ):
        """
        Initialize the ASBasinStabilityEstimator.

        :param n: Number of initial conditions (samples) to generate.
        :param ode_system: The ODE system model (ODESystem or JaxODESystem).
        :param sampler: The Sampler object to generate initial conditions.
        :param solver: The Solver object to integrate the ODE system (Solver or JaxSolver).
        :param feature_extractor: The FeatureExtractor object to extract features.
        :param cluster_classifier: The ClusterClassifier object to assign labels.
        :param as_params: The AdaptiveStudyParams object to vary the parameter.
        :param save_to: The folder where results will be saved.
        """
        self.n = n
        self.ode_system = ode_system
        self.sampler = sampler
        self.solver = solver
        self.feature_extractor = feature_extractor
        self.cluster_classifier = cluster_classifier
        self.as_params = as_params
        self.save_to = save_to

        # Add storage for parameter study results
        self.parameter_values: list[float] = []
        self.basin_stabilities: list[dict[str, float]] = []
        self.results: list[dict[str, Any]] = []

    def estimate_as_bs(
        self,
    ) -> tuple[list[float], list[dict[str, float]], list[dict[str, Any]]]:
        """
        Estimate basin stability for each parameter value.

        Uses GPU acceleration automatically when available for significant performance gains.

        :return: Tuple of (parameter_values, basin_stabilities, results)
        """
        self.parameter_values = []
        self.basin_stabilities = []
        self.results = []

        print(
            f"\nEstimating Basin Stability for parameter: {self.as_params['adaptative_parameter_name']}"
        )
        print(f"Parameter values: {self.as_params['adaptative_parameter_values']}")

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Computing device: {device}")

        for param_value in self.as_params["adaptative_parameter_values"]:
            # Update parameter using eval
            assignment = f"{self.as_params['adaptative_parameter_name']} = {param_value}"

            context: dict[str, Any] = {
                "n": self.n,
                "ode_system": self.ode_system,
                "sampler": self.sampler,
                "solver": self.solver,
                "feature_extractor": self.feature_extractor,
                "cluster_classifier": self.cluster_classifier,
            }

            eval(compile(assignment, "<string>", "exec"), context, context)

            # Evaluate the same variable/expression to get its current value
            variable_name = self.as_params["adaptative_parameter_name"]
            updated_value = eval(variable_name, context, context)

            print(f"\nCurrent {variable_name} value: {updated_value}")

            bse = BasinStabilityEstimator(
                n=context["n"],
                ode_system=context["ode_system"],
                sampler=context["sampler"],
                solver=context["solver"],
                feature_extractor=context["feature_extractor"],
                cluster_classifier=context["cluster_classifier"],
            )

            basin_stability = bse.estimate_bs()

            # Store only essential results (not the full solution to save memory)
            self.parameter_values.append(param_value)
            self.basin_stabilities.append(basin_stability)

            # Extract only necessary data from solution before storing
            if bse.solution is not None:
                solution_summary = {
                    "param_value": param_value,
                    "basin_stability": basin_stability,
                    "n_samples": len(bse.y0) if bse.y0 is not None else bse.n,
                    "labels": bse.solution.labels.copy()
                    if bse.solution.labels is not None
                    else None,
                    "bifurcation_amplitudes": (
                        bse.solution.bifurcation_amplitudes.detach().cpu().clone()
                        if bse.solution.bifurcation_amplitudes is not None
                        else None
                    ),
                }
            else:
                solution_summary = {
                    "param_value": param_value,
                    "basin_stability": basin_stability,
                    "n_samples": len(bse.y0) if bse.y0 is not None else bse.n,
                    "labels": None,
                    "bifurcation_amplitudes": None,
                }

            self.results.append(solution_summary)

            print(f"Basin Stability: {basin_stability} for parameter value {param_value}")

            # Explicitly free memory after each iteration
            del bse
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.parameter_values, self.basin_stabilities, self.results

    def save(self) -> None:
        """
        Save the basin stability results to a JSON file.
        Handles numpy arrays and Solution objects by converting them to standard Python types.
        """
        if len(self.basin_stabilities) == 0:
            raise ValueError("No results to save. Please run estimate_as_bs() first.")

        if self.save_to is None:
            raise ValueError("No path to save the results was specified.")

        full_folder = resolve_folder(self.save_to)
        file_name = generate_filename("adaptative_params_basin_stability_results", "json")
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

        adaptative_parameter_study = [
            {"param_value": param_value, "basin_of_attraction": bs}
            for param_value, bs in zip(self.parameter_values, self.basin_stabilities, strict=True)
        ]

        results: dict[str, Any] = {
            "studied_parameters": self.as_params["adaptative_parameter_name"],
            "adaptative_parameter_study": adaptative_parameter_study,
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
