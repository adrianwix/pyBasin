"""Adaptive Study Basin Stability Estimator."""

import gc
import json
import logging
import os
from typing import Any

import torch

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.sampler import Sampler
from pybasin.study_params import StudyParams
from pybasin.template_integrator import TemplateIntegrator
from pybasin.types import AdaptiveStudyResult, ErrorInfo
from pybasin.utils import NumpyEncoder, generate_filename, resolve_folder

logger = logging.getLogger(__name__)


class BasinStabilityStudy:
    """
    Basin Stability Study.
    """

    def __init__(
        self,
        n: int,
        ode_system: ODESystemProtocol,
        sampler: Sampler,
        solver: SolverProtocol,
        feature_extractor: FeatureExtractor,
        estimator: Any,
        study_params: StudyParams,
        template_integrator: TemplateIntegrator | None = None,
        save_to: str | None = "results",
        verbose: bool = False,
    ):
        """
        Initialize the Basin Stability Study.

        Sets up the estimator for a parameter study where one or more parameters are
        systematically varied across multiple values. Parameters can be in any component
        (ODE system, sampler, solver, feature extractor, or predictor).

        :param n: Number of initial conditions (samples) to generate for each parameter value.
        :param ode_system: The ODE system model (ODESystem or JaxODESystem).
        :param sampler: The Sampler object to generate initial conditions.
        :param solver: The Solver object to integrate the ODE system (Solver or JaxSolver).
        :param feature_extractor: The FeatureExtractor object to extract features from trajectories.
        :param estimator: Any sklearn-compatible estimator (classifier or clusterer).
        :param study_params: Parameter study specification (SweepStudyParams, GridStudyParams, etc.).
        :param template_integrator: Template integrator for supervised classifiers.
        :param save_to: Folder path where results will be saved, or None to disable saving.
        :param verbose: If True, show detailed logs from BasinStabilityEstimator instances.
                        If False, suppress INFO logs to reduce output clutter during parameter sweeps.
        """
        self.n = n
        self.ode_system = ode_system
        self.sampler = sampler
        self.solver = solver
        self.feature_extractor = feature_extractor
        self.estimator = estimator
        self.study_params = study_params
        self.template_integrator = template_integrator
        self.save_to = save_to
        self.verbose = verbose

        # Add storage for parameter study results
        self.labels: list[dict[str, Any]] = []
        self.basin_stabilities: list[dict[str, float]] = []
        self.results: list[AdaptiveStudyResult] = []

    @property
    def parameter_values(self) -> list[Any]:
        """Legacy access to parameter values (for backward compatibility).

        :return: List of parameter values from labels.
        """
        if not self.labels:
            return []
        first_key = next(iter(self.labels[0].keys()))
        return [label[first_key] for label in self.labels]

    def _suppress_verbose_logs(self) -> dict[str, int]:
        """Suppress verbose logs from BasinStabilityEstimator and related components.

        :return: Dictionary mapping logger names to their original log levels.
        """
        original_levels: dict[str, int] = {}

        if self.verbose:
            return original_levels

        loggers_to_suppress = [
            "pybasin.basin_stability_estimator",
            "pybasin.predictors.base",
            "pybasin.solvers.jax_solver",
        ]
        for logger_name in loggers_to_suppress:
            log = logging.getLogger(logger_name)
            original_levels[logger_name] = log.level
            log.setLevel(logging.WARNING)

        return original_levels

    def _restore_log_levels(self, original_levels: dict[str, int]) -> None:
        """Restore original log levels.

        :param original_levels: Dictionary mapping logger names to their original log levels.
        """
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)

    def estimate_as_bs(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, float]], list[AdaptiveStudyResult]]:
        """
        Estimate basin stability for each parameter combination in the study.

        Performs basin stability estimation by systematically varying parameters
        across the provided combinations. For each configuration:

        1. Applies all parameter assignments from the RunConfig
        2. Creates a new BasinStabilityEstimator instance
        3. Estimates basin stability and computes error estimates
        4. Stores results including basin stability values, errors, and solution metadata

        Uses GPU acceleration automatically when available for significant performance gains.
        Memory is explicitly freed after each iteration to prevent accumulation.

        :return: Tuple of three lists with matching indices:

                 - labels: List of label dictionaries identifying each run
                 - basin_stabilities: List of basin stability dictionaries (label -> fraction)
                 - results: List of AdaptiveStudyResult with complete information including errors
        """
        self.labels = []
        self.basin_stabilities = []
        self.results = []

        total_runs = len(self.study_params)

        logger.info("\n" + "=" * 80)
        logger.info("PARAMETER STUDY: %d runs", total_runs)
        logger.info("=" * 80)

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Computing device: %s\n", device)

        for idx, run_config in enumerate(self.study_params, 1):
            context: dict[str, Any] = {
                "n": self.n,
                "ode_system": self.ode_system,
                "sampler": self.sampler,
                "solver": self.solver,
                "feature_extractor": self.feature_extractor,
                "estimator": self.estimator,
                "template_integrator": self.template_integrator,
            }

            # Apply all parameter assignments for this run
            for assignment in run_config.assignments:
                context["_param_value"] = assignment.value
                exec_code = f"{assignment.name} = _param_value"
                eval(compile(exec_code, "<string>", "exec"), context, context)

            logger.info("\n" + "-" * 80)
            label_str = ", ".join(f"{k}={v}" for k, v in run_config.label.items())
            logger.info("[%d/%d] %s", idx, total_runs, label_str)
            logger.info("-" * 80)

            # Use sampler's n_samples if available (for CsvSampler), otherwise use self.n
            n_samples = (
                context["sampler"].n_samples
                if hasattr(context["sampler"], "n_samples")
                else context["n"]
            )

            bse = BasinStabilityEstimator(
                n=n_samples,
                ode_system=context["ode_system"],
                sampler=context["sampler"],
                solver=context["solver"],
                feature_extractor=context["feature_extractor"],
                predictor=context["estimator"],
                template_integrator=context["template_integrator"],
                feature_selector=None,
            )

            original_levels = self._suppress_verbose_logs()
            basin_stability = bse.estimate_bs()
            self._restore_log_levels(original_levels)

            # Compute errors for this parameter value
            errors = bse.get_errors()

            # Store only essential results (not the full solution to save memory)
            self.labels.append(run_config.label)
            self.basin_stabilities.append(basin_stability)

            # Get the primary parameter value for backward compatibility
            param_value = next(iter(run_config.label.values())) if run_config.label else None

            # Extract only necessary data from solution before storing
            if bse.solution is not None:
                solution_summary: AdaptiveStudyResult = {
                    "param_value": param_value,
                    "basin_stability": basin_stability,
                    "errors": errors,
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
                    "errors": errors,
                    "n_samples": len(bse.y0) if bse.y0 is not None else bse.n,
                    "labels": None,
                    "bifurcation_amplitudes": None,
                }

            self.results.append(solution_summary)

            # Format basin stability output
            bs_str = ", ".join([f"{k}: {v:.4f}" for k, v in basin_stability.items()])
            logger.info("Result: {%s}", bs_str)
            logger.info("-" * 80 + "\n")

            # Explicitly free memory after each iteration
            del bse
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.labels, self.basin_stabilities, self.results

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

        # Build study results with labels
        parameter_study = [
            {"label": label, "basin_of_attraction": bs}
            for label, bs in zip(self.labels, self.basin_stabilities, strict=True)
        ]

        # Determine studied parameters from first label keys
        studied_params = list(self.labels[0].keys()) if self.labels else []

        results: dict[str, Any] = {
            "studied_parameters": studied_params,
            "parameter_study": parameter_study,
            "region_of_interest": region_of_interest,
            "sampling_points": self.n,
            "sampling_method": self.sampler.__class__.__name__,
            "solver": self.solver.__class__.__name__,
            "estimator": self.estimator.__class__.__name__,
            "ode_system": format_ode_system(self.ode_system.get_str()),
        }

        with open(full_path, "w") as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)

        logger.info("Results saved to %s", full_path)

    def get_errors(self, param_index: int) -> dict[str, ErrorInfo]:
        """
        Get error information for basin stability estimates at a specific parameter value.

        Retrieves the pre-computed error estimates (absolute and relative standard errors)
        for all attractor labels at the specified parameter index.

        :param param_index: Index of the parameter value in the adaptive study (0-based).
        :return: Dictionary mapping each attractor label to its ErrorInfo containing e_abs and e_rel.
        :raises ValueError: If estimate_as_bs() has not been called yet.
        :raises ValueError: If param_index is out of range.
        """
        if len(self.results) == 0:
            raise ValueError("No basin stability values available. Call estimate_as_bs() first.")

        if param_index < 0 or param_index >= len(self.results):
            raise ValueError(
                f"Parameter index {param_index} out of range [0, {len(self.results) - 1}]"
            )

        return self.results[param_index]["errors"]
