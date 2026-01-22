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

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "results"

Z_THRESHOLD_OK = 2.0
Z_THRESHOLD_WARNING = 3.0


def _status_emoji(confidence: str) -> str:
    """Get status emoji based on confidence level.

    :param confidence: Confidence level string ("very_high", "high", "moderate", "low", "very_low").
    :return: Status emoji.
    """
    if confidence == "very_high":
        return "✅"  # p > 0.10 (very likely same)
    elif confidence == "high":
        return "✅"  # p > 0.05 (no significant difference)
    elif confidence == "moderate":
        return "⚠️"  # p > 0.01 (borderline)
    elif confidence == "low":
        return "❌"  # p > 0.001 (significant difference)
    else:  # very_low
        return "❌"  # p <= 0.001 (highly significant difference)


def _format_bs_with_se(bs: float, se: float) -> str:
    """Format basin stability with standard error."""
    return f"{bs:.4f} ± {se:.4f}"


def _format_ci(ci_lower: float, ci_upper: float) -> str:
    """Format confidence interval."""
    return f"[{ci_lower:.4f}, {ci_upper:.4f}]"


def comparison_table(case_id: str) -> str:
    """Render a comparison table from a JSON artifact.

    For single-point tests, renders a table with columns:
    Attractor | pyBasin BS ± SE | bSTAB BS ± SE | z-score | Status

    For parameter sweep tests, adds a Parameter column first:
    Parameter | Attractor | pyBasin BS ± SE | bSTAB BS ± SE | z-score | Status

    For unsupervised tests, adds cluster quality metrics and purity column:
    Attractor | DBSCAN | Purity | pyBasin BS ± SE | bSTAB BS ± SE | z-score | Status

    :param case_id: Case identifier (e.g., "pendulum_case1", "pendulum_case2").
    :return: Markdown table string.
    """
    json_path = ARTIFACTS_DIR / f"{case_id}_comparison.json"

    if not json_path.exists():
        return f'!!! warning "Missing Data"\n    Comparison data not found: `{case_id}_comparison.json`\n    Run tests with `--generate-artifacts` to generate.'

    with open(json_path) as f:
        data: dict[str, Any] = json.load(f)

    if "parameter_results" in data:
        return _render_parameter_sweep_table(data)
    if "overall_agreement" in data:
        return _render_unsupervised_table(data)
    return _render_single_point_table(data)


def _render_single_point_table(data: dict[str, Any]) -> str:
    """Render table for single-point comparison."""
    attractors: list[dict[str, Any]] = data.get("attractors", [])

    if not attractors:
        return '!!! warning "No Data"\n    No attractor data found in comparison.'

    lines: list[str] = [
        "| Attractor | pyBasin BS ± SE | bSTAB BS ± SE | z | p-value | 95% CI (diff) | Confidence |",
        "|-----------|-----------------|---------------|---|---------|---------------|------------|",
    ]

    for a in attractors:
        python_str = _format_bs_with_se(a["python_bs"], a["python_se"])
        matlab_str = _format_bs_with_se(a["matlab_bs"], a["matlab_se"])
        z_score: float = a["z_score"]
        p_value: float = a["p_value"]
        ci_str = _format_ci(a["ci_lower"], a["ci_upper"])
        confidence: str = a["confidence"]
        status = _status_emoji(confidence)

        lines.append(
            f"| {a['label']} | {python_str} | {matlab_str} | {z_score:.2f} | {p_value:.4f} | {ci_str} | {status}&nbsp;{confidence} |"
        )

    return "\n".join(lines)


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

    summary_lines: list[str] = [
        "**Cluster Quality Metrics:**",
        "",
        f"- Clusters found: {n_found} (expected: {n_expected})",
        f"- Overall agreement: {agreement:.1%}",
        f"- Adjusted Rand Index: {ari:.4f}",
        "",
    ]

    # Attractor table with purity info
    table_lines: list[str] = [
        "| Attractor | DBSCAN | Purity | pyBasin BS ± SE | bSTAB BS ± SE | z | p-value | Confidence |",
        "|-----------|--------|--------|-----------------|---------------|---|---------|------------|",
    ]

    for a in attractors:
        python_str = _format_bs_with_se(a["python_bs"], a["python_se"])
        matlab_str = _format_bs_with_se(a["matlab_bs"], a["matlab_se"])
        z_score: float = a["z_score"]
        p_value: float = a["p_value"]
        confidence: str = a["confidence"]
        status = _status_emoji(confidence)
        dbscan_label = a.get("dbscan_label", "-")
        purity = a.get("purity", 0.0)
        purity_str = f"{purity:.1%}"

        table_lines.append(
            f"| {a['label']} | {dbscan_label} | {purity_str} | {python_str} | {matlab_str} | {z_score:.2f} | {p_value:.4f} | {status}&nbsp;{confidence} |"
        )

    return "\n".join(summary_lines + table_lines)


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

        lines: list[str] = [
            f"#### {param_name} = {param_str}",
            "",
            "| Attractor | pyBasin BS ± SE | bSTAB BS ± SE | z | p-value | 95% CI (diff) | Confidence |",
            "|-----------|-----------------|---------------|---|---------|---------------|------------|",
        ]

        attractors: list[dict[str, Any]] = result.get("attractors", [])
        for a in attractors:
            python_str = _format_bs_with_se(a["python_bs"], a["python_se"])
            matlab_str = _format_bs_with_se(a["matlab_bs"], a["matlab_se"])
            z_score: float = a["z_score"]
            p_value: float = a["p_value"]
            ci_str = _format_ci(a["ci_lower"], a["ci_upper"])
            confidence: str = a["confidence"]
            status = _status_emoji(confidence)

            lines.append(
                f"| {a['label']} | {python_str} | {matlab_str} | {z_score:.2f} | {p_value:.4f} | {ci_str} | {status}&nbsp;{confidence} |"
            )

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


def define_env(env: Any) -> None:
    """Define macros for mkdocs-macros-plugin.

    :param env: The macro environment.
    """
    env.macro(comparison_table, "comparison_table")
    env.macro(load_snippet, "load_snippet")
