import numpy as np

from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.study_params import SweepStudyParams
from pybasin.utils import time_execution


def main():
    props = setup_pendulum_system()

    study_params = SweepStudyParams(
        name='ode_system.params["T"]',
        values=list(np.arange(0.01, 0.97, 0.05)),
        # values=list(np.arange(0.1, 1.00, 0.2)),
    )

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    estimator = props.get("estimator")
    template_integrator = props.get("template_integrator")
    assert solver is not None, "solver is required for BasinStabilityStudy"
    assert feature_extractor is not None, "feature_extractor is required for BasinStabilityStudy"
    assert estimator is not None, "estimator is required for BasinStabilityStudy"

    bse = BasinStabilityStudy(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        estimator=estimator,
        study_params=study_params,
        template_integrator=template_integrator,
        save_to="results_case2",
    )

    bse.estimate_as_bs()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_pendulum_case2.py", main)

    state_labels = {0: "θ", 1: "ω"}
    plotter = InteractivePlotter(bse, state_labels=state_labels)
    plotter.run(port=8050, debug=True)

    bse.save()
