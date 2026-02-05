#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
uv run mkdocs serve --dev-addr 127.0.0.1:8000 --watch src --watch docs --livereload
