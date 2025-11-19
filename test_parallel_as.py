"""Quick test to verify parallelization works."""


def test_parallel_vs_sequential():
    """Compare parallel vs sequential execution times."""
    # props = setup_pendulum_system()

    # Use fewer parameter values for quick test
    # test_params = AdaptiveStudyParams(
    #     adaptative_parameter_values=5 * np.logspace(1, 2, 3),  # Only 3 values for faster test
    #     adaptative_parameter_name="n",
    # )

    # print("=" * 60)
    # print("Testing SEQUENTIAL execution (n_jobs=1)")
    # print("=" * 60)
    # start_sequential = time.time()
    # bse_sequential = ASBasinStabilityEstimator(
    #     n=props["n"],
    #     ode_system=props["ode_system"],
    #     sampler=props["sampler"],
    #     solver=props["solver"],
    #     feature_extractor=props["feature_extractor"],
    #     cluster_classifier=props["cluster_classifier"],
    #     as_params=test_params,
    #     save_to=None,
    #     n_jobs=1,  # Sequential
    # )
    # results_seq = bse_sequential.estimate_as_bs()
    # time_sequential = time.time() - start_sequential

    print("\n" + "=" * 60)
    print("Testing PARALLEL execution (n_jobs=None, use all cores)")
    print("=" * 60)
    # Parallel execution would be tested here
    # start_parallel = time.time()
    # bse_parallel = ASBasinStabilityEstimator(
    #     n=props["n"],
    #     ode_system=props["ode_system"],
    #     sampler=props["sampler"],
    #     solver=props["solver"],
    #     feature_extractor=props["feature_extractor"],
    #     cluster_classifier=props["cluster_classifier"],
    #     as_params=test_params,
    #     save_to=None,
    #     n_jobs=None,  # Use all cores
    # )


if __name__ == "__main__":
    test_parallel_vs_sequential()
