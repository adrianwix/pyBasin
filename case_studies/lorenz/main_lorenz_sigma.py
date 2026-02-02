import matplotlib
import numpy as np

from pybasin.utils import time_execution

matplotlib.use("TkAgg")

from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.matplotlib_as_plotter import ASPlotter
from pybasin.study_params import SweepStudyParams


def main():
    props = setup_lorenz_system()

    study_params = SweepStudyParams(
        name='ode_system.params["sigma"]',
        values=list(np.arange(0.12, 0.1825, 0.0025)),
    )

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    cluster_classifier = props.get("cluster_classifier")
    assert solver is not None, "solver is required for BasinStabilityStudy"
    assert feature_extractor is not None, "feature_extractor is required for BasinStabilityStudy"
    assert cluster_classifier is not None, "cluster_classifier is required for BasinStabilityStudy"

    bse = BasinStabilityStudy(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=cluster_classifier,
        study_params=study_params,
        save_to="results_sigma",
    )

    bse.estimate_as_bs()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_lorenz_sigma.py", main)

    plotter = ASPlotter(bse)

    plotter.plot_basin_stability_variation()
    plotter.plot_bifurcation_diagram(dof=[0, 1, 2])

    bse.save()
