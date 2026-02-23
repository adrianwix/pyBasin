import numpy as np

from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.basin_stability_study import BasinStabilityStudy

# from pybasin.matplotlib_study_plotter import MatplotlibStudyPlotter
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.study_params import GridStudyParams
from pybasin.utils import time_execution

T_VALUES = list(np.round(np.linspace(0.1, 0.9, 5), 2))
ALPHA_VALUES = list(np.round(np.linspace(0.05, 0.3, 5), 3))
N = 500


def main() -> BasinStabilityStudy:
    props = setup_pendulum_system()

    study_params = GridStudyParams(
        **{
            'ode_system.params["T"]': T_VALUES,
            'ode_system.params["alpha"]': ALPHA_VALUES,
        }
    )

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    estimator = props.get("estimator")
    template_integrator = props.get("template_integrator")
    assert solver is not None, "solver is required for BasinStabilityStudy"
    assert feature_extractor is not None, "feature_extractor is required for BasinStabilityStudy"
    assert estimator is not None, "estimator is required for BasinStabilityStudy"

    bss = BasinStabilityStudy(
        n=N,
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        estimator=estimator,
        study_params=study_params,
        template_integrator=template_integrator,
        save_to="results_multiparams",
    )

    bss.run()

    # plotter = MatplotlibStudyPlotter(bss)
    # plotter.plot_parameter_stability(parameters=["T"])
    # plotter.plot_orbit_diagram([0, 1], parameters=["T"])
    # plt.show()

    bss.save()

    return bss


if __name__ == "__main__":
    bss = time_execution("main_pendulum_multiparams_study.py", main)

    state_labels = {0: "θ", 1: "ω"}
    plotter = InteractivePlotter(bss, state_labels=state_labels)
    plotter.run(port=8050)
