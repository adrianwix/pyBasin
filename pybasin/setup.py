from setuptools import setup, find_packages

setup(
    name="pybasin",
    description="Basin stability estimation for dynamical systems",
    version="0.1",
    packages=find_packages(),
    package_data={
        "pybasin": ["py.typed"],
    },
    install_requires=[
        # List your dependencies here
        "numpy",
        "matplotlib",
        "scikit-learn",
        "torch",
        "torchdiffeq",
        # Add other dependencies as needed
    ],
    zip_safe=False,
)
