#!/bin/bash
set -e

cd /home/adrian/code/thesis/pyBasinWorkspace

uv run austin -o benchmarks/profiling/profile.mojo python -m benchmarks.profiling.profiling_pendulum_case1_with_defaults
uv run mojo2austin benchmarks/profiling/profile.mojo benchmarks/profiling/profile.collapsed
uv run austin2speedscope benchmarks/profiling/profile.collapsed benchmarks/profiling/profile.speedscope.json

echo "Done! Open https://speedscope.app and load benchmarks/profiling/profile.speedscope.json"
