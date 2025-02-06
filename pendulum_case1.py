from typing import Tuple, List, Dict, Optional, TypedDict
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from BasinStabilityEstimator import BasinStabilityEstimator
from ODESystem import ODESystem, PendulumODE, PendulumParams
from Sampler import RandomSampler, Sampler
from Solver import SciPySolver, Solver

if __name__ == "__main__":
    # OHE = {
    #     "FP": np.array([1.0, 0.0], dtype=np.float64),
    #     "LC": np.array([0.0, 1.0], dtype=np.float64)
    # }
    OHE = {"FP": [1, 0], "LC": [0, 1]}

    # Example usage (ensure that the necessary helper functions and classes are imported):
    N = 1000
    steady_state_time = 950.0

    # Instantiate your ODE system, sampler, and solver:
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}

    ode_system = PendulumODE(params)  # for example

    sampler = RandomSampler(
        min_limits= (-np.pi + np.arcsin(params["T"] / params["K"]), -10.0), 
        max_limits= (np.pi + np.arcsin(params["T"] / params["K"]),  10.0))

    solver = SciPySolver(time_span=(0, 1000), method="RK45", rtol=1e-8)

    # For supervised clustering, you might need default templates:
    templates = np.array([OHE["FP"], OHE["LC"]], dtype=np.float64)

    bse = BasinStabilityEstimator(
        N=N,
        steady_state_time=steady_state_time,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        supervised=True,
        templates=templates  # or None to trigger default templates (if OHE is defined)
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)
    bse.plots()
    bse.save("basin_stability_results.pkl")