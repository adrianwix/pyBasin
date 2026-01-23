import numpy as np

from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.as_basin_stability_estimator import ASBasinStabilityEstimator
from pybasin.matplotlib_as_plotter import ASPlotter
from pybasin.study_params import SweepStudyParams
from pybasin.utils import time_execution


def main():
    props = setup_lorenz_system()

    study_params = SweepStudyParams(
        name="N",
        values=list(2 * np.logspace(2, 4, 50, dtype=np.int64)),
    )

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    cluster_classifier = props.get("cluster_classifier")
    assert solver is not None, "solver is required for ASBasinStabilityEstimator"
    assert feature_extractor is not None, (
        "feature_extractor is required for ASBasinStabilityEstimator"
    )
    assert cluster_classifier is not None, (
        "cluster_classifier is required for ASBasinStabilityEstimator"
    )

    bse = ASBasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=cluster_classifier,
        study_params=study_params,
        save_to="results_hyperpN",
    )

    bse.estimate_as_bs()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_lorenz_hyperpN.py", main)
    plotter = ASPlotter(bse)

    plotter.plot_basin_stability_variation(interval="log")

    bse.save()
