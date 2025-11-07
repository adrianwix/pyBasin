import numpy as np
from case_friction.setup_friction_system import setup_friction_system
from pybasin.ASBasinStabilityEstimator import ASBasinStabilityEstimator, AdaptiveStudyParams
from pybasin.ASPlotter import ASPlotter
from pybasin.utils import time_execution


def main():
    props = setup_friction_system()
    N = props["N"]
    ode_system = props["ode_system"]
    sampler = props["sampler"]
    solver = props["solver"]
    feature_extractor = props["feature_extractor"]
    knn_cluster = props["cluster_classifier"]

    as_params = AdaptiveStudyParams(
        adaptative_parameter_values=np.arange(0.85, 2.01, step=0.05),
        adaptative_parameter_name='ode_system.params["v_d"]',
    )

    bse = ASBasinStabilityEstimator(
        N=N,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=knn_cluster,
        as_params=as_params,
        save_to="results_friction_vd_study",
    )

    print("Estimating Basin Stability...")
    bse.estimate_as_bs()

    plotter = ASPlotter(bse)

    # plotter.plot_basin_stability_variation()

    plotter.plot_bifurcation_diagram()

    bse.save()


if __name__ == "__main__":
    time_execution("main_friction_v_study.py", main)
