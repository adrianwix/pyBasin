"""Build script for mypyc-compiled ODE module.

Usage:
    cd experiments/zig/zig_mypyc/odes
    uv run python setup.py build_ext --inplace

Or simply:
    uv run mypyc pendulum_ode.py
"""

from mypyc.build import mypycify
from setuptools import setup

setup(
    ext_modules=mypycify(["pendulum_ode.py"]),
)
