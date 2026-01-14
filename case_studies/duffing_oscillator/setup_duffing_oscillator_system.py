import torch
from sklearn.neighbors import KNeighborsClassifier

from case_studies.duffing_oscillator.duffing_jax_ode import DuffingJaxODE, DuffingParams
from pybasin.predictors.knn_classifier import KNNClassifier
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import JaxSolver
from pybasin.ts_torch.torch_feature_extractor import TorchFeatureExtractor
from pybasin.types import SetupProperties


def setup_duffing_oscillator_system() -> SetupProperties:
    n = 5000

    # Auto-detect device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up Duffing oscillator system on device: {device}")

    # Create ODE system instance
    params: DuffingParams = {"delta": 0.08, "k3": 1, "A": 0.2}
    ode_system = DuffingJaxODE(params)

    # Instantiate sampler, solver, feature extractor, and cluster classifier
    sampler = UniformRandomSampler(min_limits=[-1, -0.5], max_limits=[1, 1], device=device)

    solver = JaxSolver(
        time_span=(0, 1000),
        n_steps=50000,
        device=device,
        rtol=1e-8,
        atol=1e-6,
        use_cache=True,
    )

    feature_extractor = TorchFeatureExtractor(
        time_steady=900.0,
        normalize=False,
        features=None,
        features_per_state={
            0: {"maximum": None, "standard_deviation": None},
        },
    )

    classifier_initial_conditions = [
        [-0.21, 0.02],
        [1.05, 0.77],
        [-0.67, 0.02],
        [-0.46, 0.30],
        [-0.43, 0.12],
    ]

    classifier_labels = [
        "period-1 LC y_1",
        "period-1 LC y_2",
        "period-2 LC y_3",
        "period-2 LC y_4",
        "period-3 LC y_5",
    ]

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
