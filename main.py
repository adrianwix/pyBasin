#!/usr/bin/env python
# bStab_pendulum.py
"""
A single Python "canvas" script demonstrating how to compute basin stability 
for a pendulum system using a feature-based classification approach.

Requires: numpy, scipy, scikit-learn, matplotlib
"""

from typing import Tuple, List, Dict, Optional, TypedDict
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

OHE = {
    "FP": np.array([1.0, 0.0], dtype=np.float64),
    "LC": np.array([0.0, 1.0], dtype=np.float64)
}


class Params(TypedDict):
    alpha: float
    T: float
    K: float


def pendulum_ode(t: float, y: NDArray[np.float64], params: Params) -> List[float]:
    """
    Right-hand side (RHS) for the pendulum ODE.

    Parameters
    ----------
    t : float
        The current time (not explicitly used if the system is time-invariant).
    y : NDArray[np.float64], shape (2,)
        The current state, [theta, theta_dot].
    params : Params
        Dictionary of model parameters, for example {"alpha": 0.1, "T": 0.5, "K": 1.0}.

    Returns
    -------
    dydt : List[float]
        The derivatives: [dtheta/dt, dtheta_dot/dt].
    """
    alpha = params["alpha"]
    T = params["T"]
    K = params["K"]

    theta, theta_dot = y
    dtheta_dt = theta_dot
    dtheta_dot_dt = -alpha * theta_dot + T - K * np.sin(theta)

    return [dtheta_dt, dtheta_dot_dt]


def features_pendulum(
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    steady_state_time: float = 950.0
) -> NDArray[np.float64]:
    """
    Replicates the MATLAB 'features_pendulum' function in Python:
      1) Identify time indices for t > steady_state_time (steady-state).
      2) Compute Delta = |max(theta_dot) - mean(theta_dot)|.
      3) If Delta < 0.01 => [1,0] (FP), else => [0,1] (LC).

    Parameters
    ----------
    t : NDArray[np.float64]
        Time values from the integration.
    y : NDArray[np.float64], shape (len(t), 2)
        The states at each time in t.  y[:,0] = theta, y[:,1] = theta_dot
    steady_state_time : float
        Time after which we consider the system to be near steady-state.

    Returns
    -------
    X : NDArray[np.float64], shape (2, 1)
        A one-hot vector, [1,0]^T for FP or [0,1]^T for LC.
    """
    # Indices where t > steady_state_time
    idx_steady = np.where(t > steady_state_time)[0]
    if len(idx_steady) == 0:
        print("Warning: No steady state found.")
        # If we never get beyond steady_state_time, default to [1,0]
        return np.array(OHE["FP"], dtype=np.float64)

    print(f"Steady state found at t={t[idx_steady[0]]}")
    idx_start = idx_steady[0]

    # Use the second state (theta_dot) for the portion after steady_state_time
    portion = y[idx_start:, 1]
    delta = np.abs(np.max(portion) - np.mean(portion))
    print(f"Delta = {delta}")

    if delta < 0.01:
        print("Fixed Point (FP)")
        # FP (Fixed Point)
        return np.array(OHE["FP"], dtype=np.float64)
    else:
        # LC (Limit Cycle)
        return np.array(OHE["LC"], dtype=np.float64)


def cluster_assign(
    features: NDArray[np.float64],
    supervised: bool = True,
    templates: Optional[NDArray[np.float64]] = None
) -> NDArray[np.int64]:
    """
    Assign a cluster/label to each feature vector.

    In 'supervised' mode with known templates, we use k-Nearest Neighbors (k=1).
    Otherwise, we do an unsupervised KMeans with 2 clusters.

    Parameters
    ----------
    features : NDArray[np.float64], shape (N, num_features)
        Extracted features for each sample (row).
    supervised : bool
        If True, do supervised classification (requires 'templates').
        If False, do KMeans with 2 clusters.
    templates : Optional[NDArray[np.float64]]
        Labeled samples for training in the supervised scenario. 
        E.g. np.array([[1, 0], [0, 1]]) with implicit labels 0 => FP, 1 => LC.

    Returns
    -------
    assignments : NDArray[np.int64], shape (N,)
        The integer label for each feature row.
    """
    # TODO: This function looks wrong
    if supervised and (templates is not None):
        # Suppose templates is [[1,0], [0,1]] => we map them to y_template=[0,1]
        X_template = templates
        y_template = np.array([0, 1], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_template, y_template)
        assignments = knn.predict(features)
        return assignments
    else:
        # Unsupervised KMeans with 2 clusters
        kmeans = KMeans(n_clusters=2, n_init="auto")
        kmeans.fit(features)
        return kmeans.labels_.astype(np.int64)


def integrate_sample(i, Y0, params, t_span, method, steady_state_time):
    y0 = Y0[i, :]
    print(f"Integrating sample {i+1}/{len(Y0)} with initial condition {y0}")
    sol = solve_ivp(
        # Move ODE to param
        fun=lambda t, y: pendulum_ode(t, y, params),
        t_span=t_span,
        y0=y0,
        method=method,
        rtol=1e-8,
        dense_output=True
    )
    t = sol.t
    y = sol.y.T  # shape => (len(t), 2)
    X_i = features_pendulum(t, y, steady_state_time=steady_state_time)
    return X_i.flatten()  # shape => (2,)


def compute_basin_stability(
    N: int,
    time_span: Tuple[float, float],
    # TODO: Move this to features extraction. It depends on the end of time span
    # but besides that, it is not used at all.
    steady_state_time: float,
    params: Params,
    # Defines the state space limits for initial conditions (Region of Interest)
    min_limits: Tuple[float, float],
    max_limits: Tuple[float, float],
    # Clustering
    supervised: bool = True,
    templates: Optional[NDArray[np.float64]] = None,
    method: str = "RK45"
) -> Tuple[NDArray[np.int64], Dict[int, float]]:
    """
    Main function: estimate basin stability for the pendulum system.

    Steps:
    1) Sample initial conditions from a uniform distribution in [min_limits, max_limits].
    2) Integrate the ODE for each.
    3) Extract a feature vector with `features_pendulum`.
    4) Classify/cluster them with `cluster_assign`.
    5) Compute fraction in each label => "basin stability" values.

    Parameters
    ----------
    N : int
        Number of samples (initial conditions).
    time_span : Tuple[float, float]
        Time-interval for integration, (t0, tf).
    steady_state_time : float
        Time after which to measure steady-state features.
    params : Dict[str, float]
        Dictionary with pendulum parameters, e.g. {"T": 0.1, "K": 0.5}.
    min_limits : Tuple[float, float]
        Lower bounds for [theta, theta_dot].
    max_limits : Tuple[float, float]
        Upper bounds for [theta, theta_dot].
    supervised : bool
        If True, do a supervised classification with the given templates.
    templates : Optional[NDArray[np.float64]]
        A 2D array of shape (#classes, #features), e.g. for 2 classes:
        [[1, 0], [0, 1]]. Must match the "features" dimension from feature extraction.
    method : str
        ODE solver method for `solve_ivp`.

    Returns
    -------
    assignments : NDArray[np.int64], shape (N,)
        Labels for each sample (e.g. 0 => FP, 1 => LC).
    basin_stability : Dict[int, float]
        Dictionary mapping {label: fraction_of_samples_in_that_label}.
    """
    rng = np.random.default_rng()

    # Step 1: Generate random initial conditions
    # shape => (N, 2)
    thetas = rng.uniform(min_limits[0], max_limits[0], N)
    theta_dots = rng.uniform(min_limits[1], max_limits[1], N)
    Y0 = np.column_stack((thetas, theta_dots))

    # Step 2/3: Integrate and extract features
    all_features = []

    with ProcessPoolExecutor() as executor:
        all_features = list(executor.map(integrate_sample, range(
            N), [Y0]*N, [params]*N, [time_span]*N, [method]*N, [steady_state_time]*N))

    # for i in range(N):
    #     all_features.append(
    #         integrate_sample(i, Y0, params, t_span, method, t_star))

    features_array = np.vstack(all_features)  # shape => (N, 2)

    # Provide default templates if none specified and we do supervised mode
    if supervised and templates is None:
        # 2-class: FP => [1,0], LC => [0,1]
        templates = np.array([OHE["FP"], OHE["LC"]], dtype=np.float64)

    # Step 4: Classify/cluster
    assignments = cluster_assign(
        features_array, supervised=supervised, templates=templates)

    # Step 5: Compute fraction for each label
    unique_labels, counts = np.unique(assignments, return_counts=True)
    fractions = counts / float(N)
    basin_stability = dict(zip(unique_labels.tolist(), fractions.tolist()))

    return assignments, basin_stability, Y0, features_array


if __name__ == "__main__":
    # Example usage
    # We request 10_000 initial conditions, run the simulation, classify them.
    N = 1000
    # Example param dictionary
    params = {"alpha": 0.1, "T": 0.5, "K": 1.0}
    min_limits = (-np.pi + np.arcsin(params["T"] / params["K"]), -10.0)
    max_limits = (np.pi + np.arcsin(params["T"] / params["K"]),  10.0)

    assignments, basin_stab, Y0, features_array = compute_basin_stability(
        N=N,
        params=params,
        min_limits=min_limits,
        max_limits=max_limits,
        time_span=(0, 1000),
        steady_state_time=950.0,
        supervised=True,
        templates=np.array([[1, 0], [0, 1]], dtype=np.float64),
        method="RK45"
    )

    print("Assignments (first 20):", assignments[:20])
    print("Basin stability:", basin_stab)

    plt.figure(figsize=(10, 6))

    # 1) Bar plot for the basin stability values
    plt.subplot(2, 2, 1)
    labels, values = zip(*basin_stab.items())
    plt.bar(labels, values, color=["#1f77b4", "#ff7f0e"])
    plt.xticks(labels)
    plt.ylabel("Fraction of samples")
    plt.title("Basin Stability")

    # 2) State space scatter plot: class-labeled initial conditions
    plt.subplot(2, 2, 2)
    for label in np.unique(assignments):
        idx = assignments == label
        plt.scatter(
            Y0[idx, 0], Y0[idx, 1],
            s=5, alpha=0.5, label=f"Class {label}"
        )
    plt.title("Initial Conditions in State Space")
    plt.xlabel("theta")
    plt.ylabel("theta_dot")
    plt.legend()

    # 3) Feature space scatter plot with classifier results
    plt.subplot(2, 2, 3)
    for label in np.unique(assignments):
        idx = assignments == label
        class_name = "Fixed Point" if label == 0 else "Limit Cycle"
        plt.scatter(
            features_array[idx, 0], features_array[idx, 1],
            s=5, alpha=0.5, label=class_name
        )
    plt.title("Feature Space with Classifier Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # 4) Empty plot for future use
    plt.subplot(2, 2, 4)
    plt.title("Future Plot")

    plt.tight_layout()
    plt.show()
