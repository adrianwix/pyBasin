#!/usr/bin/env bash
set -e

echo "Running case studies with default settings..."
echo ""

uv run python -m case_studies.run_case_studies_with_defaults

echo ""
echo "Done. Results saved to artifacts/results/"
