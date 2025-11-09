from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.cluster_classifier import DBSCANCluster
from pybasin.plotter import Plotter

if __name__ == "__main__":
    # Use the setup function to get all system components
    setup = setup_duffing_oscillator_system()

    cluster_classifier = DBSCANCluster(eps=0.08)

    bse = BasinStabilityEstimator(
        n=setup["n"],
        ode_system=setup["ode_system"],
        sampler=setup["sampler"],
        solver=setup["solver"],
        feature_extractor=setup["feature_extractor"],
        cluster_classifier=cluster_classifier,
        save_to="results_unsupervised",
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    plotter = Plotter(bse)

    plotter.plot_bse_results()

    bse.save()
