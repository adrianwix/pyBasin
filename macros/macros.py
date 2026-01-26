"""MkDocs macros for case study documentation.

This module provides macros for rendering comparison tables from JSON artifacts
and loading code snippets from source files.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeArgument=false

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "results"

Z_THRESHOLD_OK = 2.0
Z_THRESHOLD_WARNING = 3.0


def _format_bs_with_se(bs: float, se: float) -> str:
    """Format basin stability with standard error."""
    return f"{bs:.4f} ± {se:.4f}"


def comparison_table(case_id: str) -> str:
    """Render a comparison table from a JSON artifact.

    For single-point tests, renders a table with columns:
    Attractor | pyBasin BS ± SE | bSTAB BS ± SE | F1 | MCC

    For parameter sweep tests, adds a Parameter column first:
    Parameter | Attractor | pyBasin BS ± SE | bSTAB BS ± SE | F1 | MCC

    For unsupervised tests, adds cluster quality metrics and purity column:
    Attractor | DBSCAN | Purity | pyBasin BS ± SE | bSTAB BS ± SE | F1 | MCC

    For paper validation tests (no ground truth labels), shows confidence intervals:
    Attractor | pyBasin BS ± SE | Paper BS ± SE | Difference | 95% CI | Status

    Also shows overall Macro F1 and MCC in a summary section.

    :param case_id: Case identifier (e.g., "pendulum_case1", "pendulum_case2").
    :return: Markdown table string.
    """
    json_path = ARTIFACTS_DIR / f"{case_id}_comparison.json"

    if not json_path.exists():
        return f'!!! warning "Missing Data"\n    Comparison data not found: `{case_id}_comparison.json`\n    Run tests with `--generate-artifacts` to generate.'

    with open(json_path) as f:
        data: dict[str, Any] = json.load(f)

    # Detect paper validation case: attractors have f1_score == 0.0 (N/A)
    is_paper_validation = False
    if "attractors" in data:
        attractors: list[dict[str, Any]] = data["attractors"]
        if attractors and attractors[0].get("f1_score", 1.0) == 0.0:
            is_paper_validation = True
    elif "parameter_results" in data:
        param_results: list[dict[str, Any]] = data["parameter_results"]
        if param_results and "attractors" in param_results[0]:
            first_attractors: list[dict[str, Any]] = param_results[0]["attractors"]
            if first_attractors and first_attractors[0].get("f1_score", 1.0) == 0.0:
                is_paper_validation = True

    if is_paper_validation:
        if "parameter_results" in data:
            return _render_paper_validation_sweep_table(data)
        return _render_paper_validation_table(data)

    if "parameter_results" in data:
        return _render_parameter_sweep_table(data)
    if "overall_agreement" in data:
        return _render_unsupervised_table(data)
    return _render_single_point_table(data)


def _render_paper_validation_table(data: dict[str, Any]) -> str:
    """Render table for paper validation (statistical comparison)."""
    attractors: list[dict[str, Any]] = data.get("attractors", [])

    if not attractors:
        return '!!! warning "No Data"\n    No attractor data found in comparison.'

    # Table header
    table_lines: list[str] = [
        "| Attractor | pyBasin BS ± SE | Paper BS ± SE | Difference | 95% CI | Status |",
        "|-----------|-----------------|---------------|------------|--------|--------|",
    ]

    for a in attractors:
        python_bs: float = a["python_bs"]
        python_se: float = a["python_se"]
        matlab_bs: float = a["matlab_bs"]
        matlab_se: float = a["matlab_se"]

        python_str = _format_bs_with_se(python_bs, python_se)
        matlab_str = _format_bs_with_se(matlab_bs, matlab_se)

        # Compute difference and 95% confidence interval
        diff = python_bs - matlab_bs
        # For 95% confidence, z = 1.96
        combined_se = (python_se**2 + matlab_se**2) ** 0.5
        ci_margin = 1.96 * combined_se

        # Check if difference is within CI (difference should contain 0)
        within_ci = abs(diff) <= ci_margin
        status = "✓" if within_ci else "✗"

        diff_str = f"{diff:+.4f}"
        ci_str = f"±{ci_margin:.4f}"

        table_lines.append(
            f"| {a['label']} | {python_str} | {matlab_str} | {diff_str} | {ci_str} | {status} |"
        )

    return "\n".join(table_lines)


def _render_single_point_table(data: dict[str, Any]) -> str:
    """Render table for single-point comparison."""
    attractors: list[dict[str, Any]] = data.get("attractors", [])

    if not attractors:
        return '!!! warning "No Data"\n    No attractor data found in comparison.'

    # Get overall metrics
    macro_f1 = data.get("macro_f1", 0.0)
    mcc = data.get("matthews_corrcoef", 0.0)

    # Summary metrics
    summary_lines: list[str] = [
        "**Overall Classification Quality:**",
        "",
        f"- Macro F1-score: {macro_f1:.4f}",
        f"- Matthews Correlation Coefficient: {mcc:.4f}",
        "",
    ]

    # Attractor table
    table_lines: list[str] = [
        "| Attractor | pyBasin BS ± SE | bSTAB BS ± SE | F1 |",
        "|-----------|-----------------|---------------|-----|",
    ]

    for a in attractors:
        python_str = _format_bs_with_se(a["python_bs"], a["python_se"])
        matlab_str = _format_bs_with_se(a["matlab_bs"], a["matlab_se"])
        f1_score: float = a["f1_score"]

        table_lines.append(f"| {a['label']} | {python_str} | {matlab_str} | {f1_score:.4f} |")

    return "\n".join(summary_lines + table_lines)


def _render_unsupervised_table(data: dict[str, Any]) -> str:
    """Render table for unsupervised clustering comparison."""
    attractors: list[dict[str, Any]] = data.get("attractors", [])

    if not attractors:
        return '!!! warning "No Data"\n    No attractor data found in comparison.'

    # Cluster quality metrics summary
    n_found = data.get("n_clusters_found", 0)
    n_expected = data.get("n_clusters_expected", 0)
    agreement = data.get("overall_agreement", 0.0)
    ari = data.get("adjusted_rand_index", 0.0)
    macro_f1 = data.get("macro_f1", 0.0)
    mcc = data.get("matthews_corrcoef", 0.0)

    summary_lines: list[str] = [
        "**Cluster Quality Metrics:**",
        "",
        f"- Clusters found: {n_found} (expected: {n_expected})",
        f"- Overall agreement: {agreement:.1%}",
        f"- Adjusted Rand Index: {ari:.4f}",
        f"- Macro F1-score: {macro_f1:.4f}",
        f"- Matthews Correlation Coefficient: {mcc:.4f}",
        "",
    ]

    # Attractor table with purity info
    table_lines: list[str] = [
        "| Attractor | DBSCAN | Purity | pyBasin BS ± SE | bSTAB BS ± SE | F1 |",
        "|-----------|--------|--------|-----------------|---------------|-----|",
    ]

    for a in attractors:
        python_str = _format_bs_with_se(a["python_bs"], a["python_se"])
        matlab_str = _format_bs_with_se(a["matlab_bs"], a["matlab_se"])
        f1_score: float = a["f1_score"]
        dbscan_label = a.get("dbscan_label", "-")
        purity = a.get("purity", 0.0)
        purity_str = f"{purity:.1%}"

        table_lines.append(
            f"| {a['label']} | {dbscan_label} | {purity_str} | {python_str} | {matlab_str} | {f1_score:.4f} |"
        )

    return "\n".join(summary_lines + table_lines)


def _render_paper_validation_sweep_table(data: dict[str, Any]) -> str:
    """Render table for paper validation parameter sweep (statistical comparison)."""
    parameter_results: list[dict[str, Any]] = data.get("parameter_results", [])

    if not parameter_results:
        return '!!! warning "No Data"\n    No parameter data found in comparison.'

    param_name: str = data.get("parameter_name", "Parameter")
    sections: list[str] = []

    for result in parameter_results:
        param_value: float | None = result.get("parameter_value")
        param_str = f"{param_value:.4f}" if param_value is not None else "-"

        lines: list[str] = [
            f"#### {param_name} = {param_str}",
            "",
            "| Attractor | pyBasin BS ± SE | Paper BS ± SE | Difference | 95% CI | Status |",
            "|-----------|-----------------|---------------|------------|--------|--------|",
        ]

        attractors: list[dict[str, Any]] = result.get("attractors", [])
        for a in attractors:
            python_bs: float = a["python_bs"]
            python_se: float = a["python_se"]
            matlab_bs: float = a["matlab_bs"]
            matlab_se: float = a["matlab_se"]

            python_str = _format_bs_with_se(python_bs, python_se)
            matlab_str = _format_bs_with_se(matlab_bs, matlab_se)

            # Compute difference and 95% confidence interval
            diff = python_bs - matlab_bs
            # For 95% confidence, z = 1.96
            combined_se = (python_se**2 + matlab_se**2) ** 0.5
            ci_margin = 1.96 * combined_se

            # Check if difference is within CI (difference should contain 0)
            within_ci = abs(diff) <= ci_margin
            status = "✓" if within_ci else "✗"

            diff_str = f"{diff:+.4f}"
            ci_str = f"±{ci_margin:.4f}"

            lines.append(
                f"| {a['label']} | {python_str} | {matlab_str} | {diff_str} | {ci_str} | {status} |"
            )

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _render_parameter_sweep_table(data: dict[str, Any]) -> str:
    """Render table for parameter sweep comparison."""
    parameter_results: list[dict[str, Any]] = data.get("parameter_results", [])

    if not parameter_results:
        return '!!! warning "No Data"\n    No parameter data found in comparison.'

    param_name: str = data.get("parameter_name", "Parameter")
    sections: list[str] = []

    for result in parameter_results:
        param_value: float | None = result.get("parameter_value")
        param_str = f"{param_value:.4f}" if param_value is not None else "-"
        macro_f1 = result.get("macro_f1", 0.0)
        mcc = result.get("matthews_corrcoef", 0.0)

        lines: list[str] = [
            f"#### {param_name} = {param_str}",
            "",
            f"**Macro F1:** {macro_f1:.4f} | **MCC:** {mcc:.4f}",
            "",
            "| Attractor | pyBasin BS ± SE | bSTAB BS ± SE | F1 |",
            "|-----------|-----------------|---------------|-----|",
        ]

        attractors: list[dict[str, Any]] = result.get("attractors", [])
        for a in attractors:
            python_str = _format_bs_with_se(a["python_bs"], a["python_se"])
            matlab_str = _format_bs_with_se(a["matlab_bs"], a["matlab_se"])
            f1_score: float = a["f1_score"]

            lines.append(f"| {a['label']} | {python_str} | {matlab_str} | {f1_score:.4f} |")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def load_snippet(spec: str) -> str:
    """Load a code snippet from a source file.

    :param spec: Specification in format "path/to/file.py::function_name"
                 Path should be relative to the workspace root.
    :return: Markdown-formatted code block with the extracted function.
    """
    try:
        file_path_str, func_name = spec.split("::")
    except ValueError:
        return f'!!! error "Invalid Format"\n    Expected format: `path/to/file.py::function_name`\n    Got: `{spec}`'

    workspace_root = Path(__file__).parent.parent
    file_path = workspace_root / file_path_str

    if not file_path.exists():
        return f'!!! error "File Not Found"\n    Could not find file: `{file_path_str}`'

    try:
        source_code = file_path.read_text()
        tree = ast.parse(source_code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                lines = source_code.splitlines()
                start_line = node.lineno - 1
                end_line = node.end_lineno if node.end_lineno else len(lines)

                function_code = "\n".join(lines[start_line:end_line])

                return f"```python\n{function_code}\n```"

        return f'!!! warning "Function Not Found"\n    Could not find function `{func_name}` in `{file_path_str}`'

    except SyntaxError as e:
        return f'!!! error "Syntax Error"\n    Failed to parse `{file_path_str}`: {e}'
    except Exception as e:
        return f'!!! error "Error"\n    Failed to load snippet: {e}'


BENCHMARK_RESULTS_DIR = Path(__file__).parent.parent / "benchmarks" / "end_to_end" / "results"
SOLVER_COMPARISON_RESULTS_DIR = (
    Path(__file__).parent.parent / "benchmarks" / "solver_comparison" / "results"
)


def solver_comparison_table() -> str:
    """Render solver comparison table from CSV data.

    :return: Markdown table string.
    """
    csv_path = SOLVER_COMPARISON_RESULTS_DIR / "solver_comparison.csv"

    if not csv_path.exists():
        return '!!! warning "Missing Data"\n    Solver comparison data not found. Run `uv run python benchmarks/solver_comparison/compare_matlab_vs_python.py` to generate.'

    df = pd.read_csv(csv_path)

    n_values = sorted(df["N"].unique())
    sections: list[str] = []

    for n in n_values:
        n_data = df[df["N"] == n].sort_values("mean_time")
        matlab_row = n_data[n_data["solver"] == "MATLAB ode45"]
        matlab_time = matlab_row["mean_time"].values[0] if len(matlab_row) > 0 else None

        table_lines: list[str] = [
            f"### N = {n:,}",
            "",
            "| Solver | Device | Time (s) | Std Dev | vs MATLAB |",
            "|--------|--------|----------:|--------:|----------:|",
        ]

        for _, row in n_data.iterrows():
            if matlab_time is not None:
                speedup = matlab_time / row["mean_time"]
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "-"
            table_lines.append(
                f"| {row['solver']} | {row['device'].upper()} | {row['mean_time']:.2f} | ±{row['std_time']:.2f} | {speedup_str} |"
            )

        sections.append("\n".join(table_lines))

    return "\n\n".join(sections)


def solver_matlab_speedup_table() -> str:
    """Render speedup vs MATLAB table from CSV data.

    :return: Markdown table string.
    """
    csv_path = SOLVER_COMPARISON_RESULTS_DIR / "solver_comparison.csv"

    if not csv_path.exists():
        return '!!! warning "Missing Data"\n    Solver comparison data not found.'

    df = pd.read_csv(csv_path)

    n_values = sorted(df["N"].unique())

    table_lines: list[str] = [
        "| N | Solver | Device | Time (s) | vs MATLAB |",
        "|--:|--------|--------|----------:|----------:|",
    ]

    for n in n_values:
        n_data = df[df["N"] == n]
        matlab_row = n_data[n_data["solver"] == "MATLAB ode45"]

        if matlab_row.empty:
            continue

        matlab_time = matlab_row["mean_time"].values[0]

        for _, row in (
            n_data[n_data["solver"] != "MATLAB ode45"].sort_values("mean_time").iterrows()
        ):
            speedup = matlab_time / row["mean_time"]
            direction = "faster" if speedup > 1 else "slower"
            speedup_str = f"{speedup:.2f}x {direction}"
            table_lines.append(
                f"| {n:,} | {row['solver']} | {row['device'].upper()} | {row['mean_time']:.2f} | {speedup_str} |"
            )

        table_lines.append(f"| {n:,} | MATLAB ode45 | CPU | {matlab_time:.2f} | *baseline* |")

    return "\n".join(table_lines)


def benchmark_comparison_table() -> str:
    """Render benchmark comparison table from CSV data.

    :return: Markdown table string.
    """
    csv_path = BENCHMARK_RESULTS_DIR / "end_to_end_comparison.csv"

    if not csv_path.exists():
        return '!!! warning "Missing Data"\n    Benchmark data not found. Run `uv run python benchmarks/end_to_end/compare_matlab_vs_python.py` to generate.'

    df = pd.read_csv(csv_path)

    n_values = sorted(df["N"].unique())

    table_lines: list[str] = [
        "| N | MATLAB (s) | Python CPU (s) | Python CUDA (s) | CPU vs MATLAB | GPU vs MATLAB |",
        "|--:|----------:|---------------:|----------------:|--------------:|--------------:|",
    ]

    for n in n_values:
        n_data = df[df["N"] == n]

        matlab_row = n_data[n_data["implementation"] == "MATLAB"]
        cpu_row = n_data[n_data["implementation"] == "Python CPU"]
        cuda_row = n_data[n_data["implementation"] == "Python CUDA"]

        matlab_time = matlab_row["mean_time"].values[0] if len(matlab_row) > 0 else float("nan")
        cpu_time = cpu_row["mean_time"].values[0] if len(cpu_row) > 0 else float("nan")
        cuda_time = cuda_row["mean_time"].values[0] if len(cuda_row) > 0 else float("nan")

        cpu_speedup = matlab_time / cpu_time if cpu_time > 0 else float("nan")
        gpu_speedup = matlab_time / cuda_time if cuda_time > 0 else float("nan")

        table_lines.append(
            f"| {n:,} | {matlab_time:.2f} | {cpu_time:.2f} | {cuda_time:.2f} | {cpu_speedup:.1f}x | {gpu_speedup:.1f}x |"
        )

    return "\n".join(table_lines)


def benchmark_scaling_analysis() -> str:
    """Render scaling analysis summary from benchmark data.

    :return: Markdown summary string.
    """
    csv_path = BENCHMARK_RESULTS_DIR / "end_to_end_comparison.csv"

    if not csv_path.exists():
        return '!!! warning "Missing Data"\n    Benchmark data not found.'

    df = pd.read_csv(csv_path)

    implementations = ["MATLAB", "Python CPU", "Python CUDA"]
    results: list[str] = [
        "| Implementation | Scaling | Exponent α | R² |",
        "|----------------|---------|------------|-----|",
    ]

    for impl in implementations:
        impl_data = df[df["implementation"] == impl].sort_values("N")
        if len(impl_data) < 3:
            continue

        n_vals = impl_data["N"].values.astype(float)
        t_vals = impl_data["mean_time"].values

        log_n = np.log(n_vals)
        log_t = np.log(t_vals)
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(log_n, log_t)
        alpha = slope
        alpha_ci = 1.96 * std_err
        r2 = r_value**2

        if alpha < 0.15:
            complexity = "O(1)"
        elif abs(alpha - 1.0) < 0.15:
            complexity = "O(N)"
        elif abs(alpha - 2.0) < 0.15:
            complexity = "O(N²)"
        else:
            complexity = f"O(N^{alpha:.2f})"

        results.append(f"| {impl} | {complexity} | {alpha:.2f} ± {alpha_ci:.2f} | {r2:.3f} |")

    return "\n".join(results)


def define_env(env: Any) -> None:
    """Define macros for mkdocs-macros-plugin.

    :param env: The macro environment.
    """
    env.macro(comparison_table, "comparison_table")
    env.macro(load_snippet, "load_snippet")
    env.macro(benchmark_comparison_table, "benchmark_comparison_table")
    env.macro(benchmark_scaling_analysis, "benchmark_scaling_analysis")
    env.macro(solver_comparison_table, "solver_comparison_table")
    env.macro(solver_matlab_speedup_table, "solver_matlab_speedup_table")
