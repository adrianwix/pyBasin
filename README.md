# pyBasin

**Basin stability estimation for dynamical systems**

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

pyBasin is a Python library for estimating basin stability in dynamical systems. This is a modern port of the MATLAB bSTAB library with additional features including adaptive sampling and neural network-based classification.

## 🎯 Features

- **Basin Stability Estimation**: Calculate the probability that a system ends up in specific attractors
- **Adaptive Sampling**: Intelligent sampling strategies that focus computational resources on uncertain regions
- **Multiple Solvers**: Support for various ODE solvers including neural ODE
- **Visualization Tools**: Built-in plotting utilities for basin stability results
- **Extensible Architecture**: Easy to add custom feature extractors and classifiers
- **Type-Safe**: Full type annotations with py.typed marker

## 📦 Installation

### From PyPI (when published)

```bash
pip install pybasin
```

### From Source

```bash
# Clone the repository
git clone https://github.com/adrianwix/pyBSTAB.git
cd pyBasinWorkspace

# Create virtual environment with UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
uv pip install -e .

# Or install with all optional dependencies
uv pip install -e ".[all]"
```

## 🚀 Quick Start

```python
import numpy as np
from pybasin import BasinStabilityEstimator, ODESystem

# Define your dynamical system
class MySystem(ODESystem):
    def dynamics(self, t, state):
        x, y = state
        dx = -x + y
        dy = -y - x**3
        return np.array([dx, dy])
    
    def classify_attractor(self, solution):
        final_state = solution.y[:, -1]
        return 0 if np.linalg.norm(final_state) < 0.1 else 1

# Create estimator and run
system = MySystem()
estimator = BasinStabilityEstimator(system)
bounds = [(-2, 2), (-2, 2)]
results = estimator.estimate(bounds, n_samples=1000)

print(f"Basin stability: {results.basin_stability}")
```

## 📚 Documentation

Full documentation is available at: **[https://adrianwix.github.io/pyBSTAB/](https://adrianwix.github.io/pyBSTAB/)**

Or build locally:

```bash
uv pip install -e ".[docs]"
mkdocs serve
```

Then visit http://localhost:8000

## 🧪 Case Studies

This repository includes validated case studies from the original bSTAB paper:

| Case Study | Location | Description |
|------------|----------|-------------|
| **Duffing Oscillator** | `case_studies/duffing_oscillator/` | Forced oscillator with bistability |
| **Lorenz System** | `case_studies/lorenz/` | Classic chaotic attractor |
| **Pendulum** | `case_studies/pendulum/` | Forced pendulum with bifurcations |
| **Friction System** | `case_studies/friction/` | Mechanical system with friction |

Run a case study:

```bash
uv run python case_studies/duffing_oscillator/main_supervised.py
```

## 📁 Project Structure

```
pyBasinWorkspace/
├── src/pybasin/          # 📦 Main library code
├── case_studies/         # 🔬 Research case studies
│   ├── duffing_oscillator/
│   ├── lorenz/
│   ├── pendulum/
│   └── friction/
├── tests/                # ✅ Unit and integration tests
│   └── integration/      # Validation against MATLAB
├── docs/                 # 📖 Documentation source
├── artifacts/            # 📊 Generated figures and results
├── scripts/              # 🛠️ Helper scripts
└── notebooks/            # 📓 Jupyter examples
```

## 🧑‍💻 Development

### Setup Development Environment

```bash
# Install all dependencies including dev tools
uv pip install -e ".[all]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pybasin

# Run only integration tests
pytest tests/integration/

# Run specific case study test
pytest tests/integration/test_duffing.py
```

### Code Quality

```bash
# Format code
black src/ tests/ case_studies/

# Lint
ruff check src/ tests/ case_studies/

# Type check
mypy src/
```

## 📊 Validation

pyBasin is validated against the original MATLAB bSTAB implementation. Integration tests in `tests/integration/` compare basin stability values for all case studies.

## 🎓 Academic Context

This library is part of a bachelor thesis on basin stability estimation for dynamical systems. It ports and extends the functionality of the MATLAB bSTAB library with modern Python practices and additional features.

## 🔗 Related Projects

- **bSTAB (MATLAB)**: Original implementation - [GitHub](https://github.com/original/bSTAB)

## 📝 Citation

If you use pyBasin in your research, please cite:

```bibtex
@software{pybasin2025,
  author = {Wix, Adrian},
  title = {pyBasin: Basin Stability Estimation for Dynamical Systems},
  year = {2025},
  url = {https://github.com/adrianwix/pyBSTAB}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please see the [Contributing Guide](docs/development/contributing.md) for details.

## 📧 Contact

- **Author**: Adrian Wix
- **Repository**: [https://github.com/adrianwix/pyBSTAB](https://github.com/adrianwix/pyBSTAB)
- **Issues**: [https://github.com/adrianwix/pyBSTAB/issues](https://github.com/adrianwix/pyBSTAB/issues)

---

**Note**: This is an active development project. The API may change before the 1.0 release.
