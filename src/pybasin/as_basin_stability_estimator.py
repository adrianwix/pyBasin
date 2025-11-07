"""Adaptive Study Basin Stability Estimator."""

import json
import os
from typing import TypedDict

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.cluster_classifier import ClusterClassifier
from pybasin.feature_extractor import FeatureExtractor
from pybasin.ode_system import ODESystem
from pybasin.sampler import Sampler
from pybasin.solver import Solver
from pybasin.utils import NumpyEncoder, generate_filename, resolve_folder


class AdaptiveStudyParams(TypedDict):
    adaptative_parameter_values: list
    adaptative_parameter_name: str


class ASBasinStabilityEstimator:
    """
    Adaptive Study Basin Stability Estimator.
    """

    def __init__(
        self,
        N: int,
        ode_system: ODESystem,
        sampler: Sampler,
        solver: Solver,
        feature_extractor: FeatureExtractor,
        cluster_classifier: ClusterClassifier,
        as_params: AdaptiveStudyParams,
        save_to: str | None = "results",
    ):
        """
        Initialize the BasinStabilityEstimator.

        :param N: Number of initial conditions (samples) to generate.
        :param steady_state_time: Time after which steady-state features are extracted.
        :param ode_system: The ODE system model.
        :param sampler: The Sampler object to generate initial conditions.
        :param solver: The Solver object to integrate the ODE system.
        :param cluster_classifier: The ClusterClassifier object to assign labels.
        :param as_params: The AdaptiveStudyParams object to vary the parameter.
        :param save_to: The folder where results will be saved.
        """
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
        self.results = []

    def estimate_as_bs(self) -> dict[int, float]:
        """Returns list of basin stability values for each parameter value"""
        self.parameter_values = []
        self.basin_stabilities = []

        print(
            f"\nEstimating Basin Stability for parameter: {self.as_params['adaptative_parameter_name']}"
        )

        print(f"\nParameter values: {self.as_params['adaptative_parameter_values']}")

        for param_value in self.as_params["adaptative_parameter_values"]:
            # Update parameter using eval
            assignment = f"{self.as_params['adaptative_parameter_name']} = {param_value}"

            context = {
                "N": self.N,
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
                N=context["N"],
                ode_system=context["ode_system"],
                sampler=context["sampler"],
                solver=context["solver"],
                feature_extractor=context["feature_extractor"],
                cluster_classifier=context["cluster_classifier"],
            )

            basin_stability = bse.estimate_bs()

            # Store results
            self.parameter_values.append(param_value)
            self.basin_stabilities.append(basin_stability)
            self.results.append(
                {
                    "param_value": param_value,
                    "basin_stability": basin_stability,
                    "solution": bse.solution,
                }
            )

            print(f"Basin Stability: {basin_stability} for parameter value {param_value}")

        return self.parameter_values, self.basin_stabilities, self.results

    def save(self):
        """
        Save the basin stability results to a JSON file.
        Handles numpy arrays and Solution objects by converting them to standard Python types.
        """
        if self.basin_stabilities is None:
            raise ValueError("No results to save. Please run estimate_as_bs() first.")

        if self.save_to is None:
            raise ValueError("No path to save the results was specified.")

        full_folder = resolve_folder(self.save_to)
        file_name = generate_filename("adaptative_params_basin_stability_results", "json")
        full_path = os.path.join(full_folder, file_name)

        def format_ode_system(ode_str: str) -> list:
            lines = ode_str.strip().split("\n")
            formatted_lines = [" ".join(line.split()) for line in lines]
            return formatted_lines

        region_of_interest = " X ".join(
            [
                f"[{min_val}, {max_val}]"
                for min_val, max_val in zip(self.sampler.min_limits, self.sampler.max_limits)
            ]
        )

        adaptative_parameter_study = [
            {"param_value": param_value, "basin_of_attraction": bs}
            for param_value, bs in zip(self.parameter_values, self.basin_stabilities)
        ]

        results = {
            "studied_parameters": self.as_params["adaptative_parameter_name"],
            "adaptative_parameter_study": adaptative_parameter_study,
            "region_of_interest": region_of_interest,
            "sampling_points": self.N,
            "sampling_method": self.sampler.__class__.__name__,
            "solver": self.solver.__class__.__name__,
            "cluster_classifier": self.cluster_classifier.__class__.__name__,
            "ode_system": format_ode_system(self.ode_system.get_str()),
        }

        with open(full_path, "w") as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)

        print(f"Results saved to {full_path}")
