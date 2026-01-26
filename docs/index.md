# pyBasin

**Basin stability estimation for dynamical systems**

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

pyBasin is a Python library for estimating basin stability in dynamical systems. It's a port of the MATLAB bSTAB library with additional features including adaptive sampling and neural network-based classification.

## Features

- **Basin Stability Estimation**: Calculate the probability that a system ends up in a specific attractor
- **Adaptive Sampling**: Intelligent sampling strategies that focus on uncertain regions
- **Multiple Solvers**: Support for various ODE solvers including neural ODE
- **Visualization Tools**: Built-in plotting utilities for basin stability results
- **Extensible**: Easy to add custom feature extractors and classifiers

## Installation

```bash
pip install pybasin
```

For development:

```bash
# Clone the repository
git clone https://github.com/adrianwix/pyBSTAB.git
cd pyBasinWorkspace

# Install with UV
uv add -e ".[dev,docs]"
```

## Quick Start

```python
from pybasin import BasinStabilityEstimator, ODESystem
import numpy as np

# Define your dynamical system
class MySystem(ODESystem):
    def dynamics(self, t, state):
        x, y = state
        dx = -x + y
        dy = -y - x**3
        return np.array([dx, dy])

    def classify_attractor(self, solution):
        # Classify final state
        final_state = solution.y[:, -1]
        if np.linalg.norm(final_state) < 0.1:
            return 0  # Attractor 1
        return 1  # Attractor 2

# Create estimator
system = MySystem()
estimator = BasinStabilityEstimator(system)

# Define sampling region
bounds = [(-2, 2), (-2, 2)]  # x and y bounds

# Estimate basin stability
results = estimator.estimate(bounds, n_samples=1000)

print(f"Basin stability: {results.basin_stability}")
print(f"Attractor distribution: {results.attractor_counts}")
```

## Documentation

Full documentation is available at [https://adrianwix.github.io/pyBSTAB/](https://adrianwix.github.io/pyBSTAB/)

## Case Studies

This repository includes several case studies from the original bSTAB paper:

- **Duffing Oscillator**: Forced oscillator with two attractors
- **Lorenz System**: Classic chaotic system
- **Pendulum**: Forced pendulum with bifurcations
- **Friction System**: System with friction effects

See the `case_studies/` directory for implementations.

## Project Structure

```
pyBasinWorkspace/
├── src/pybasin/          # Main library code
├── case_studies/         # Research case studies
├── tests/                # Unit and integration tests
├── docs/                 # Documentation source
├── artifacts/            # Generated figures and results
└── notebooks/            # Jupyter notebook examples
```

## Development

### Setup

```bash
# Install all dependencies including dev tools
uv add -e ".[all]"
```

### Running Tests

```bash
pytest
```

### Building Documentation

```bash
mkdocs serve  # Local preview
mkdocs build  # Build static site
```

## Related Projects

- **bSTAB**: Original MATLAB implementation - [GitHub](https://github.com/original/bSTAB)

## Citation

If you use pyBasin in your research, please cite:

```bibtex
@software{pybasin2025,
  author = {Wix, Adrian},
  title = {pyBasin: Basin Stability Estimation for Dynamical Systems},
  year = {2025},
  url = {https://github.com/adrianwix/pyBSTAB}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the bSTAB MATLAB library
- Part of a bachelor thesis on basin stability estimation
