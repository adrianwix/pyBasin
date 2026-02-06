# pyright: basic
"""
Compare MATLAB and Python benchmark results for basin stability estimation.

Loads benchmark results from both implementations and creates comparison plots
grouped by N (number of samples). Uses CPSME styling for thesis-quality plots.
"""

import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit

from pybasin.thesis_utils import THESIS_PALETTE, thesis_export


def load_matlab_results(matlab_json_path: Path) -> dict:
    with open(matlab_json_path) as f:
        data = json.load(f)
    return data


def load_python_results(python_json_path: Path) -> dict:
    with open(python_json_path) as f:
        data = json.load(f)
    return data


def extract_matlab_data(matlab_results: dict) -> pd.DataFrame:
    benchmarks = matlab_results["benchmarks"]

    rows = []
    for bench in benchmarks:
        rows.append(
            {
                "N": bench["N"],
                "mean_time": bench["mean_time"],
                "std_time": bench["std_time"],
                "min_time": bench["min_time"],
                "max_time": bench["max_time"],
                "implementation": "MATLAB",
            }
        )

    return pd.DataFrame(rows)


def extract_python_data(python_results: dict) -> pd.DataFrame:
    benchmarks = python_results["benchmarks"]

    rows = []
    for bench in benchmarks:
        params = bench["params"]
        n = params["n"]
        stats = bench["stats"]

        rows.append(
            {
                "N": n,
                "mean_time": stats["mean"],
                "std_time": stats["stddev"],
                "min_time": stats["min"],
                "max_time": stats["max"],
                "implementation": f"Python {params['device'].upper()}",
            }
        )

    return pd.DataFrame(rows)


def create_comparison_plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    n_values = sorted(df["N"].unique())
    implementations = ["Python CPU", "Python CUDA", "MATLAB"]
    colors = {
        "Python CPU": THESIS_PALETTE[0],
        "Python CUDA": THESIS_PALETTE[2],
        "MATLAB": THESIS_PALETTE[1],
    }

    x = np.arange(len(n_values))
    width = 0.25

    for i, impl in enumerate(implementations):
        impl_data = df[df["implementation"] == impl].set_index("N")
        means = np.array(
            [impl_data.loc[n, "mean_time"] if n in impl_data.index else 0.0 for n in n_values],
            dtype=float,
        )
        stds = np.array(
            [impl_data.loc[n, "std_time"] if n in impl_data.index else 0.0 for n in n_values],
            dtype=float,
        )

        ax.bar(
            x + i * width,
            means,
            width,
            label=impl,
            color=colors[impl],
            yerr=stds,
            capsize=3,
        )

    ax.set_xlabel("Number of Samples (N)")
    ax.set_ylabel("Mean Time (seconds)")
    ax.set_title("Basin Stability Computation: Python CPU vs Python GPU vs MATLAB")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{n:,}" for n in n_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    thesis_export(fig, output_path.name, output_path.parent)
    print(f"Comparison plot saved to: {output_path}")


def print_comparison_table(df: pd.DataFrame) -> None:
    print("\n=== Benchmark Comparison ===")
    df_sorted = df.sort_values(["N", "implementation"])
    print(df_sorted.to_string(index=False))

    print("\n=== Speedup Analysis ===")
    for n in sorted(df["N"].unique()):
        n_data = cast(pd.DataFrame, df[df["N"] == n])
        if len(n_data) == 2:
            matlab_df = cast(pd.DataFrame, n_data[n_data["implementation"] == "MATLAB"])
            matlab_time = float(matlab_df["mean_time"].iloc[0])
            python_df = cast(pd.DataFrame, n_data[n_data["implementation"] != "MATLAB"])
            python_row = python_df.iloc[0]
            python_time = float(python_row["mean_time"])
            speedup = matlab_time / python_time

            print(
                f"N={n:6d}: MATLAB={matlab_time:6.2f}s, {python_row['implementation']}={python_time:6.2f}s, "
                f"Speedup={speedup:.2f}x"
            )


def analyze_scaling(df: pd.DataFrame) -> dict[str, dict]:
    """
    Analyze time complexity scaling for each implementation.

    Fits T = c * N^alpha (power law) and T = a * N * log(N) + b (linearithmic)
    to determine which model best describes the scaling behavior.
    """
    results: dict[str, dict] = {}

    implementations = ["MATLAB", "Python CPU", "Python CUDA"]

    print("\n=== Time Complexity Analysis ===")

    for impl in implementations:
        impl_data = cast(pd.DataFrame, df[df["implementation"] == impl]).sort_values(by="N")
        if len(impl_data) < 3:
            continue

        n_vals = np.asarray(impl_data["N"].values, dtype=float)
        t_vals = np.asarray(impl_data["mean_time"].values, dtype=float)

        log_n = np.log(n_vals)
        log_t = np.log(t_vals)
        result = scipy_stats.linregress(log_n, log_t)
        slope = result.slope
        r_value = result.rvalue
        std_err = result.stderr
        alpha = slope
        alpha_ci = 1.96 * std_err
        r2_power = r_value**2

        def linear_model(n: np.ndarray, a: float, b: float) -> np.ndarray:
            return a * n + b

        def nlogn_model(n: np.ndarray, a: float, b: float) -> np.ndarray:
            return a * n * np.log(n) + b

        try:
            popt_linear, _ = curve_fit(linear_model, n_vals, t_vals, p0=[1e-3, 1])
            residuals_linear = t_vals - linear_model(n_vals, *popt_linear)
            ss_res_linear = float(np.sum(residuals_linear**2))
            ss_tot = float(np.sum((t_vals - np.mean(t_vals)) ** 2))
            r2_linear = 1 - ss_res_linear / ss_tot
        except RuntimeError:
            r2_linear = 0.0

        try:
            popt_nlogn, _ = curve_fit(nlogn_model, n_vals, t_vals, p0=[1e-5, 1])
            residuals_nlogn = t_vals - nlogn_model(n_vals, *popt_nlogn)
            ss_res_nlogn = float(np.sum(residuals_nlogn**2))
            r2_nlogn = 1 - ss_res_nlogn / ss_tot
        except RuntimeError:
            r2_nlogn = 0.0

        if alpha < 0.15:
            complexity = "O(1) - constant"
        elif r2_linear > 0.99 and abs(alpha - 1.0) < 0.1:
            complexity = "O(N) - linear"
        elif r2_nlogn > r2_linear and r2_nlogn > 0.99:
            complexity = "O(N log N) - linearithmic"
        elif abs(alpha - 1.0) < 0.15:
            complexity = "O(N) - linear"
        elif abs(alpha - 2.0) < 0.15:
            complexity = "O(N²) - quadratic"
        else:
            complexity = f"O(N^{alpha:.2f})"

        results[impl] = {
            "alpha": alpha,
            "alpha_ci": alpha_ci,
            "r2_power": r2_power,
            "r2_linear": r2_linear,
            "r2_nlogn": r2_nlogn,
            "complexity": complexity,
        }

        print(f"\n{impl}:")
        print(f"  Power law fit: T ∝ N^{alpha:.3f} ± {alpha_ci:.3f} (R² = {r2_power:.4f})")
        print(f"  Linear fit R²: {r2_linear:.4f}")
        print(f"  N log N fit R²: {r2_nlogn:.4f}")
        print(f"  → Scaling: {complexity}")

    return results


def create_scaling_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Create log-log plot showing scaling behavior with fitted lines."""
    fig, ax = plt.subplots(figsize=(10, 7))

    implementations = ["MATLAB", "Python CPU", "Python CUDA"]
    colors = {
        "Python CPU": THESIS_PALETTE[0],
        "Python CUDA": THESIS_PALETTE[2],
        "MATLAB": THESIS_PALETTE[1],
    }
    markers = {"Python CPU": "o", "Python CUDA": "s", "MATLAB": "^"}

    for impl in implementations:
        impl_data = cast(pd.DataFrame, df[df["implementation"] == impl]).sort_values(by="N")
        if len(impl_data) < 3:
            continue

        n_vals = np.asarray(impl_data["N"].values, dtype=float)
        t_vals = np.asarray(impl_data["mean_time"].values, dtype=float)
        t_std = np.asarray(impl_data["std_time"].values, dtype=float)

        ax.errorbar(
            n_vals,
            t_vals,
            yerr=t_std,
            fmt=markers[impl],
            color=colors[impl],
            label=impl,
            capsize=3,
            markersize=8,
        )

        log_n = np.log(n_vals)
        log_t = np.log(t_vals)
        result = scipy_stats.linregress(log_n, log_t)
        slope = result.slope
        intercept = result.intercept

        n_fit = np.logspace(np.log10(n_vals.min()), np.log10(n_vals.max()), 100)
        t_fit = np.exp(intercept) * n_fit**slope
        ax.plot(
            n_fit, t_fit, "--", color=colors[impl], alpha=0.7, label=f"{impl}: O(N^{slope:.2f})"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Samples (N)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Scaling Analysis: Time vs N (log-log)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    thesis_export(fig, output_path.name, output_path.parent)
    print(f"Scaling plot saved to: {output_path}")


def main():
    results_dir = Path(__file__).parent / "results"
    docs_assets_dir = (
        Path(__file__).parent.parent.parent / "docs" / "assets" / "benchmarks" / "end_to_end"
    )
    docs_assets_dir.mkdir(parents=True, exist_ok=True)

    matlab_json = results_dir / "matlab_basin_stability_estimator_scaling.json"
    python_json = results_dir / "python_basin_stability_estimator_scaling.json"

    if not matlab_json.exists():
        print(f"MATLAB results not found at: {matlab_json}")
        return

    if not python_json.exists():
        print(f"Python results not found at: {python_json}")
        return

    print(f"Loading MATLAB results from: {matlab_json}")
    matlab_results = load_matlab_results(matlab_json)

    print(f"Loading Python results from: {python_json}")
    python_results = load_python_results(python_json)

    matlab_df = extract_matlab_data(matlab_results)
    python_df = extract_python_data(python_results)

    combined_df = pd.concat([matlab_df, python_df], ignore_index=True)

    print_comparison_table(combined_df)

    analyze_scaling(combined_df)

    output_plot = docs_assets_dir / "end_to_end_comparison.png"
    create_comparison_plot(combined_df, output_plot)

    scaling_plot = docs_assets_dir / "end_to_end_scaling.png"
    create_scaling_plot(combined_df, scaling_plot)

    output_csv = results_dir / "end_to_end_comparison.csv"
    combined_df.to_csv(output_csv, index=False)
    print(f"Comparison data saved to: {output_csv}")


if __name__ == "__main__":
    main()
