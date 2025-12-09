"""
Lorenz Hyperparameter Study: Tolerance Variation

This script computes the sensitivity of basin stability values against
the choice of the ODE integration tolerance (rtol).

This is the Python equivalent of the MATLAB script:
bSTAB-M/case_lorenz/main_lorenz_hyperpTol.m

The study varies the relative tolerance (rtol) from 1e-3 to 1e-8 to observe
how the basin stability estimates change with different integration accuracies.
"""

import numpy as np

from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.as_basin_stability_estimator import AdaptiveStudyParams, ASBasinStabilityEstimator
from pybasin.as_plotter import ASPlotter


def main():
    """
    Main function to run the Lorenz hyperparameter tolerance study.

    This study investigates how basin stability estimates vary with
    different ODE integration tolerances (rtol).
    """
    print("=" * 80)
    print("Lorenz System: Hyperparameter Study - Integration Tolerance")
    print("=" * 80)

    # Use the standard setup function to configure the Lorenz system
    props = setup_lorenz_system()

    # Override the sample size for this specific study
    # Matches MATLAB: props.roi.N = 20000
    n = 20000

    # Define the adaptive study parameters
    # Varies relative tolerance from 1e-3 to 1e-8
    # Matches MATLAB: props.ap_study.ap_values = [1.0e-03, ..., 1.0e-08]
    rtol_values = np.array([1.0e-03, 1.0e-04, 1.0e-05, 1.0e-06, 1.0e-07, 1.0e-08])

    # Create adaptive study parameters
    as_params = AdaptiveStudyParams(
        adaptative_parameter_values=rtol_values,
        adaptative_parameter_name="solver.rtol",
    )

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    cluster_classifier = props.get("cluster_classifier")
    assert solver is not None, "solver is required for ASBasinStabilityEstimator"
    assert feature_extractor is not None, (
        "feature_extractor is required for ASBasinStabilityEstimator"
    )
    assert cluster_classifier is not None, (
        "cluster_classifier is required for ASBasinStabilityEstimator"
    )

    # Initialize the Adaptive Study Basin Stability Estimator
    bse = ASBasinStabilityEstimator(
        n=n,
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        cluster_classifier=cluster_classifier,
        as_params=as_params,
        save_to="results_hyperpTol",
    )

    print("\n" + "=" * 80)
    print("Starting Basin Stability Estimation...")
    print("=" * 80)

    # Run the adaptive study
    bse.estimate_as_bs()

    print("\n" + "=" * 80)
    print("Basin Stability Estimation Complete")
    print("=" * 80)

    # Display results summary
    print("\nResults Summary:")
    print("-" * 80)
    print(f"{'rtol':<12} {'butterfly1':<15} {'butterfly2':<15} {'unbounded':<15}")
    print("-" * 80)

    for param_val, bs_dict in zip(bse.parameter_values, bse.basin_stabilities, strict=True):
        bs_b1 = bs_dict.get("butterfly1", 0.0)
        bs_b2 = bs_dict.get("butterfly2", 0.0)
        bs_ub = bs_dict.get("unbounded", 0.0)
        print(f"{param_val:<12.1e} {bs_b1:<15.6f} {bs_b2:<15.6f} {bs_ub:<15.6f}")

    # Save results to JSON
    print("\n" + "=" * 80)
    print("Saving Results...")
    print("=" * 80)
    bse.save()

    # Generate plots
    print("\n" + "=" * 80)
    print("Generating Plots...")
    print("=" * 80)

    plotter = ASPlotter(bse)

    # Plot basin stability variation vs rtol
    # Matches MATLAB: plot_bs_hyperparameter_study(props, res_tab, false)
    plotter.plot_basin_stability_variation(interval="log")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print("\nResults and figures saved to: artifacts/results/results_hyperpTol/")


if __name__ == "__main__":
    main()
