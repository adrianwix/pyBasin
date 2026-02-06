#!/bin/bash
# Generate benchmark comparison plots for documentation
# These plots use CPSME styling for thesis-quality output
set -e

cd /home/adrian/code/thesis/pyBasinWorkspace

echo "=== Generating End-to-End Benchmark Plots ==="
uv run python -m benchmarks.end_to_end.compare_matlab_vs_python

echo ""
echo "=== Generating Solver Comparison Plots ==="
uv run python -m benchmarks.solver_comparison.compare_matlab_vs_python

echo ""
echo "=== Done! ==="
echo "Plots generated in:"
echo "  - docs/assets/benchmarks/end_to_end/"
echo "  - docs/assets/benchmarks/solver_comparison/"
