#!/bin/bash
# Check Python docstrings for formatting issues (Google/NumPy style vs Sphinx)

cd "$(dirname "$0")/.." || exit 1

# Use provided path or default to src/pybasin/
path="${1:-src/pybasin/}"
shift 2>/dev/null || true

uv run python .github/skills/python-documentation-writer/check_docstrings.py "$path" "$@"
