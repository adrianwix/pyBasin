# pyright: basic
"""
Compare MATLAB and Python solver benchmark results.

Loads benchmark results from both implementations and creates comparison plots
grouped by N (number of samples). Uses CPSME styling for thesis-quality plots.
"""

import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from pybasin.docs_plots_utils import thesis_export as docs_export
from pybasin.thesis_plots_utils import (
    THESIS_FULL_WIDTH_CM,
    THESIS_PALETTE,
    THESIS_SINGLE_HEIGHT_CM,
    configure_thesis_style,
    thesis_export,
)

configure_thesis_style()

THESIS_ARTIFACTS_DIR: Path = (
    Path(__file__).parent.parent.parent / "artifacts" / "benchmarks" / "solver_comparison"
)


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
                "N": bench["n"],
                "mean_time": bench["mean_time"],
                "std_time": bench["std_time"],
                "min_time": bench["min_time"],
                "max_time": bench["max_time"],
                "solver": "MATLAB ode45",
                "device": "cpu",
            }
        )

    return pd.DataFrame(rows)


def extract_python_data(python_results: dict) -> pd.DataFrame:
    benchmarks = python_results["benchmarks"]

    rows = []
    for bench in benchmarks:
        name = bench["name"]
        params = bench["params"]
        stats = bench["stats"]

        if "jax_diffrax" in name:
            solver = "JAX/Diffrax"
        elif "torchdiffeq" in name:
            solver = "torchdiffeq"
        elif "torchode" in name:
            solver = "torchode"
        elif "scipy" in name:
            solver = "scipy"
        else:
            solver = "unknown"

        device = params.get("device", "cuda" if "torchode" in name else "cpu")

        rows.append(
            {
                "N": params["n"],
                "mean_time": stats["mean"],
                "std_time": stats["stddev"],
                "min_time": stats["min"],
                "max_time": stats["max"],
                "solver": solver,
                "device": device,
            }
        )

    return pd.DataFrame(rows)


SOLVER_COLORS = {
    "MATLAB ode45": THESIS_PALETTE[1],  # red
    "JAX/Diffrax": THESIS_PALETTE[0],  # dark blue
    "torchdiffeq": THESIS_PALETTE[5],  # grey (was teal blue, too similar to torchode)
    "torchode": THESIS_PALETTE[2],  # teal green
    "scipy": THESIS_PALETTE[4],  # medium blue
}


def _build_comparison_figure(df: pd.DataFrame, n: int) -> Figure:
    """Build the comparison bar chart figure for a given N value."""
    fig, ax = plt.subplots(figsize=(10, 6))
    n_data = cast(pd.DataFrame, df[df["N"] == n]).copy()

    n_data["label"] = n_data.apply(lambda row: f"{row['solver']} ({row['device'].upper()})", axis=1)
    n_data = n_data.sort_values(by="mean_time", ascending=True)

    # Scale down torchode CUDA for N=100000 only (divide by 3)
    plot_times = n_data["mean_time"].copy()
    plot_std = n_data["std_time"].copy()
    scale_factor = 3
    torchode_cuda_mask = (n_data["solver"] == "torchode") & (n_data["device"] == "cuda")
    should_scale = (n == 100000) & torchode_cuda_mask

    plot_times.loc[should_scale] = plot_times.loc[should_scale] / scale_factor
    plot_std.loc[should_scale] = plot_std.loc[should_scale] / scale_factor

    # Update labels for scaled bars
    labels = n_data["label"].tolist()
    for i, (idx, _row) in enumerate(n_data.iterrows()):
        if should_scale.loc[cast(int, idx)]:
            labels[i] = f"{labels[i]} (÷{scale_factor})"

    ax.barh(
        range(len(n_data)),
        plot_times,
        xerr=plot_std,
        capsize=3,
        color=[SOLVER_COLORS.get(s, "#333333") for s in n_data["solver"]],
    )

    ax.set_yticks(range(len(n_data)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"ODE Solver Benchmark: N = {n:,} samples")
    ax.grid(True, alpha=0.3, axis="x")

    # Add time labels
    for i, (_, row) in enumerate(n_data.iterrows()):
        display_time = plot_times.iloc[i]
        actual_time = row["mean_time"]
        ax.text(
            display_time + plot_std.iloc[i] + 0.3,
            i,
            f"{actual_time:.2f}s",
            va="center",
            fontsize=9,
        )

    return fig


def create_comparison_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Create one comparison plot per N value."""
    n_values = sorted(df["N"].unique())

    for n in n_values:
        output_path = output_dir / f"solver_comparison_n{n}.png"

        # Export PNG for docs
        fig_docs = _build_comparison_figure(df, n)
        docs_export(fig_docs, output_path.name, output_dir)
        plt.close(fig_docs)
        print(f"Plot saved to: {output_path}")

        # Export PDF for thesis
        fig_thesis = _build_comparison_figure(df, n)
        pdf_name = f"solver_comparison_n{n}.pdf"
        thesis_export(
            fig_thesis,
            pdf_name,
            THESIS_ARTIFACTS_DIR,
            width=THESIS_FULL_WIDTH_CM,
            height=THESIS_SINGLE_HEIGHT_CM,
        )
        plt.close(fig_thesis)


def print_comparison_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("SOLVER BENCHMARK COMPARISON")
    print("=" * 80)

    for n in sorted(df["N"].unique()):
        print(f"\n--- N = {n:,} ---")
        n_data = cast(pd.DataFrame, df[df["N"] == n]).sort_values(by="mean_time")

        fastest_time = float(n_data["mean_time"].min())

        for _, row in n_data.iterrows():
            speedup = row["mean_time"] / fastest_time
            speedup_str = f"({speedup:.1f}x slower)" if speedup > 1.01 else "(fastest)"
            print(
                f"  {row['solver']:15} ({row['device']:4}): "
                f"{row['mean_time']:8.2f} ± {row['std_time']:.2f}s  {speedup_str}"
            )

    print("\n" + "=" * 80)
    print("SPEEDUP vs MATLAB ode45")
    print("=" * 80)

    for n in sorted(df["N"].unique()):
        n_data = cast(pd.DataFrame, df[df["N"] == n])
        matlab_data = cast(pd.DataFrame, n_data[n_data["solver"] == "MATLAB ode45"])

        if len(matlab_data) == 0:
            continue

        matlab_time = float(matlab_data["mean_time"].iloc[0])
        print(f"\n--- N = {n:,} (MATLAB baseline: {matlab_time:.2f}s) ---")

        python_data = cast(pd.DataFrame, n_data[n_data["solver"] != "MATLAB ode45"]).sort_values(
            by="mean_time"
        )
        for _, row in python_data.iterrows():
            speedup = matlab_time / row["mean_time"]
            direction = "faster" if speedup > 1 else "slower"
            print(
                f"  {row['solver']:15} ({row['device']:4}): "
                f"{row['mean_time']:8.2f}s  → {abs(speedup):.2f}x {direction}"
            )


def main() -> None:
    results_dir = Path(__file__).parent / "results"
    docs_assets_dir = (
        Path(__file__).parent.parent.parent / "docs" / "assets" / "benchmarks" / "solver_comparison"
    )
    docs_assets_dir.mkdir(parents=True, exist_ok=True)
    THESIS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    matlab_json = results_dir / "matlab_benchmark_results.json"
    python_json = results_dir / "python_benchmark_results.json"

    if not matlab_json.exists():
        print(f"MATLAB results not found at: {matlab_json}")
        print("Run the MATLAB benchmark first:")
        print("  matlab -nodisplay -nosplash -r \"run('benchmark_matlab_ode45.m'); exit\"")
        return

    if not python_json.exists():
        print(f"Python results not found at: {python_json}")
        print("Run the Python benchmark first:")
        print(
            "  uv run pytest benchmarks/solver_comparison/benchmark_solver_comparison.py "
            "--benchmark-only --benchmark-json=benchmarks/solver_comparison/results/benchmark_results.json"
        )
        return

    print(f"Loading MATLAB results from: {matlab_json}")
    matlab_results = load_matlab_results(matlab_json)

    print(f"Loading Python results from: {python_json}")
    python_results = load_python_results(python_json)

    matlab_df = extract_matlab_data(matlab_results)
    python_df = extract_python_data(python_results)

    combined_df = pd.concat([matlab_df, python_df], ignore_index=True)

    print_comparison_table(combined_df)

    create_comparison_plots(combined_df, docs_assets_dir)

    output_csv = results_dir / "solver_comparison.csv"
    combined_df.to_csv(output_csv, index=False)
    print(f"\nComparison data saved to: {output_csv}")


if __name__ == "__main__":
    main()
