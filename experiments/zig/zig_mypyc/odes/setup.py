"""Build script for mypyc-compiled ODE module.

Usage:
    cd experiments/zig/zig_mypyc/odes
    uv run python setup.py build_ext --inplace

Or simply:
    uv run mypyc pendulum_ode.py
"""

from setuptools import setup

from mypyc.build import mypycify

setup(
    ext_modules=mypycify(["pendulum_ode.py"]),
)
