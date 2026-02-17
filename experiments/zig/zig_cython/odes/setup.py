"""Build script for Cython ODE modules.

Usage:
    cd experiments/zig_cffi/odes
    python setup.py build_ext --inplace

Or simply:
    cythonize -i pendulum_ode.pyx
"""

from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "pendulum_ode",
        ["pendulum_ode.pyx"],
    ),
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
