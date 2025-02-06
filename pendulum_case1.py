import numpy as np
from BasinStabilityEstimator import BasinStabilityEstimator
from ODESystem import PendulumODE, PendulumParams
from Sampler import RandomSampler
from Solver import SciPySolver

from FeatureExtractor import PendulumFeatureExtractor, PendulumOHE

if __name__ == "__main__":
    # Example usage (ensure that the necessary helper functions and classes are imported):
    N = 1000

    # Instantiate your ODE system, sampler, and solver:
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}

    ode_system = PendulumODE(params)  # for example

    sampler = RandomSampler(
        min_limits= (-np.pi + np.arcsin(params["T"] / params["K"]), -10.0), 
        max_limits= (np.pi + np.arcsin(params["T"] / params["K"]),  10.0))

    solver = SciPySolver(time_span=(0, 1000), method="RK45", rtol=1e-8)

    feature_extractor = PendulumFeatureExtractor(time_steady=950)

    # For supervised clustering, you might need default templates:
    templates = np.array([PendulumOHE["FP"], PendulumOHE["LC"]], dtype=np.float64)

    bse = BasinStabilityEstimator(
        N=N,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        # TODO: Move to cluster class
        supervised=True,
        templates=templates  # or None to trigger default templates (if OHE is defined)
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)
    bse.plots()
    bse.save("basin_stability_results.pkl")