from pybasin.ClusterClassifier import DBSCANCluster
from pybasin.Plotter import Plotter
from pybasin.BasinStabilityEstimator import BasinStabilityEstimator
from case_study_duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)

if __name__ == "__main__":
    # Use the setup function to get all system components
    setup = setup_duffing_oscillator_system()

    cluster_classifier = DBSCANCluster(eps=0.08)

    bse = BasinStabilityEstimator(
        name="duffing_oscillator_unsupervised",
        N=setup["N"],
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
