#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
uv run mkdocs gh-deploy
