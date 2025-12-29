import time

from pybasin.feature_extractors.torch_feature_calculators import DEFAULT_TORCH_FC_PARAMETERS
from pybasin.feature_extractors.torch_feature_extractor import TorchFeatureExtractor
from pybasin.feature_extractors.torch_feature_processors import extract_features_parallel

from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.solution import Solution


def main():
    props = setup_pendulum_system()

    # Run solver to get trajectories
    print("Setting up and solving ODE system...")
    y0 = props["sampler"].sample(props["n"])
    t, y = props.get("solver").integrate(props["ode_system"], y0)

    print(f"Trajectory shape: {y.shape}")
    print(f"Time span: {t[0]} to {t[-1]}")

    # Extract steady state part - matching BasinStabilityEstimator default (85% of time span)
    time_span = props.get("solver").time_span
    time_steady = time_span[0] + 0.85 * (time_span[1] - time_span[0])
    print(f"Calculated time_steady: {time_steady}")

    steady_start_idx = int((time_steady - time_span[0]) / (time_span[1] - time_span[0]) * len(t))

    y_steady = y[steady_start_idx:]
    print(f"Steady state trajectory shape: {y_steady.shape}")
    print(f"Steady state starts at t={t[steady_start_idx]:.2f}")

    # Move to CPU for extract_features_parallel
    y_steady_cpu = y_steady.cpu()

    # Time the direct extract_features_parallel call on STEADY STATE
    print("\n" + "=" * 60)
    print(f"Timing extract_features_parallel on STEADY STATE (~{y_steady.shape[0]} timesteps)")
    print("=" * 60)

    start_time = time.perf_counter()
    features = extract_features_parallel(
        y_steady_cpu,
        DEFAULT_TORCH_FC_PARAMETERS,
        n_workers=None,  # Use all CPUs
    )
    elapsed = time.perf_counter() - start_time

    print(f"\nDirect extract_features_parallel time: {elapsed:.4f}s")
    print(f"Number of feature types extracted: {len(features)}")

    # Show feature shapes
    print("\nFeature shapes (first 5):")
    for fname, tensor in list(features.items())[:5]:
        print(f"  {fname}: {tensor.shape}")

    # Now time the TorchFeatureExtractor approach
    print("\n" + "=" * 60)
    print("Timing TorchFeatureExtractor.extract_features")
    print("=" * 60)

    # Create solution object
    solution = Solution(
        initial_condition=y0,
        time=t,
        y=y,
        model_params=props["ode_system"].params,
    )

    # Create extractor with same config
    extractor = TorchFeatureExtractor(
        time_steady=time_steady,
        features=DEFAULT_TORCH_FC_PARAMETERS,
        normalize=True,
        device="cpu",
        n_jobs=None,
    )

    start_time = time.perf_counter()
    features_extractor = extractor.extract_features(solution)
    elapsed_extractor = time.perf_counter() - start_time

    print(f"\nTorchFeatureExtractor.extract_features time: {elapsed_extractor:.4f}s")
    print(f"Extracted features shape: {features_extractor.shape}")
    print(f"Slowdown factor: {elapsed_extractor / elapsed:.2f}x")

    # Show feature names
    print("\nFirst 10 feature names:")
    for name in extractor.feature_names[:10]:
        print(f"  {name}")


if __name__ == "__main__":
    main()
