import torch
from sklearn.neighbors import KNeighborsClassifier

from case_studies.friction.friction_jax_ode import FrictionJaxODE, FrictionParams
from pybasin.feature_extractors.jax_feature_extractor import JaxFeatureExtractor
from pybasin.predictors.knn_classifier import KNNClassifier
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import JaxSolver
from pybasin.types import SetupProperties


def setup_friction_system() -> SetupProperties:
    n = 5000  # Number of samples as in setup_friction.m

    # Auto-detect device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up friction system on device: {device}")

    # Parameters from setup_friction.m
    params: FrictionParams = {
        "v_d": 1.5,  # Driving velocity
        "xi": 0.05,  # Damping ratio
        "musd": 2.0,  # Ratio static to dynamic friction coefficient
        "mud": 0.5,  # Dynamic coefficient of friction
        "muv": 0.0,  # Linear strengthening parameter
        "v0": 0.5,  # Reference velocity for exponential decay
    }

    ode_system = FrictionJaxODE(params)

    # Sampling limits from setup_friction.m
    sampler = UniformRandomSampler(
        min_limits=[-2.0, 0.0],  # [disp, vel]
        max_limits=[2.0, 2.0],  # [disp, vel]
        device=device,
    )

    # Time integration parameters from setup_friction.m
    solver = JaxSolver(
        time_span=(0, 500),
        n_steps=500,
        device=device,
        rtol=1e-8,
        atol=1e-6,
        use_cache=True,
    )

    # Feature extraction (using last 100 time units as in setup_friction.m)
    feature_extractor = JaxFeatureExtractor(time_steady=400.0, normalize=False)

    # Template initial conditions for classification from setup_friction.m
    classifier_initial_conditions = [
        [0.1, 0.1],  # Fixed point (FP)
        [2.0, 2.0],  # Limit cycle (LC)
    ]

    classifier_labels = ["FP", "LC"]

    # KNN classifier as specified in setup_friction.m
    knn = KNeighborsClassifier(n_neighbors=1)

    knn_cluster = KNNClassifier(
        classifier=knn,
        template_y0=classifier_initial_conditions,
        labels=classifier_labels,
        ode_params=params,
    )

    return {
        "n": n,
        "ode_system": ode_system,
        "sampler": sampler,
        "solver": solver,
        "feature_extractor": feature_extractor,
        "cluster_classifier": knn_cluster,
    }
