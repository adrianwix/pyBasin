# pyBasin

**Basin stability estimation for dynamical systems**

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

pyBasin is a Python library for estimating basin stability in dynamical systems. This is a modern port of the MATLAB bSTAB library with additional features including parameter studies and neural network-based classification.

## 🎯 Features

- **Basin Stability Estimation**: Calculate the probability that a system ends up in specific attractors
- **Multiple Sampling Strategies**: Grid sampling, uniform random sampling, and Gaussian sampling
- **Multiple Solvers**: Support for JAX-based and PyTorch ODE solvers with GPU acceleration
- **Feature Extraction**: Extract features from trajectories for classification (JAX-based and Tsfresh)
- **Supervised & Unsupervised Classification**: KNN clustering and other sklearn classifiers
- **Visualization Tools**: Built-in plotting utilities for basin stability results
- **Extensible Architecture**: Easy to add custom feature extractors and classifiers
- **Type-Safe**: Full type annotations with py.typed marker
- **High Performance**: GPU acceleration and caching for efficient computation

## 📦 Installation

**Requirements:** Python 3.12 or higher, [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### From PyPI (when published)

```bash
pip install pybasin
```

### From Source

```bash
# Clone the repository
git clone https://github.com/adrianwix/pyBasin.git
cd pyBasinWorkspace

# Install Python 3.12 (reads version from .python-version file)
uv python install

# Install all dependencies (creates .venv automatically)
uv sync --all-groups

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## 🚀 Quick Start

### What is Basin Stability?

Basin stability measures the probability that a dynamical system, starting from a random initial condition, will converge to a specific attractor. For example, in a system with two attractors (a fixed point and a limit cycle), basin stability tells us what fraction of the phase space leads to each attractor.

### Tutorial: Analyzing a Bistable System

Let's analyze a simple 2D system with two competing attractors:

```python
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.classifiers.knncluster import KNNCluster
from pybasin.feature_extractors.jax.jax_feature_extractor import JaxFeatureExtractor
from pybasin.jax_ode_system import JaxODESystem
from pybasin.sampler import GridSampler
from pybasin.solvers import JaxSolver
```

#### Step 1: Define Your Dynamical System

First, we define the ODE system. This system has a fixed point at the origin and can exhibit limit cycle behavior depending on initial conditions:

```python
class MySystem(JaxODESystem):
    """A bistable system with a fixed point and limit cycle."""
    def dynamics(self, t, state, params):
        x, y = state
        dx = -x + y
        dy = -y - x**3
        return np.array([dx, dy])

ode_system = MySystem(params={})
```

#### Step 2: Choose a Sampling Strategy

We need to sample initial conditions across the phase space. pyBasin supports multiple sampling strategies:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

# Option A: Grid sampling (uniform coverage)
sampler = GridSampler(
    min_limits=[-2, -2],  # Lower bounds for [x, y]
    max_limits=[2, 2],    # Upper bounds for [x, y]
    device=device
)

# Option B: Random uniform sampling
# from pybasin.sampler import UniformRandomSampler
# sampler = UniformRandomSampler(min_limits=[-2, -2], max_limits=[2, 2], device=device)

# Option C: Gaussian sampling
# from pybasin.sampler import GaussianSampler
# sampler = GaussianSampler(min_limits=[-2, -2], max_limits=[2, 2], std_factor=0.3, device=device)
```

#### Step 3: Configure the ODE Solver

The solver integrates each initial condition forward in time:

```python
solver = JaxSolver(
    time_span=(0, 100),   # Integrate from t=0 to t=100
    n_steps=1000,         # Number of time steps (Δt = 0.1)
    device=device,
    rtol=1e-8,            # Relative tolerance for adaptive stepping
    atol=1e-6,            # Absolute tolerance
    use_cache=True        # Cache results for efficiency
)
```

#### Step 4: Extract Features from Trajectories

We extract features from the steady-state portion of trajectories to characterize attractors:

```python
feature_extractor = JaxFeatureExtractor(
    time_steady=90.0,  # Only use t > 90 (steady state)
    state_to_features={
        0: [],              # No features from x (state 0)
        1: ["log_delta"]    # Log of velocity variation in y
    }
    # The log_delta feature helps distinguish:
    # - Fixed points: small delta → large negative log_delta
    # - Limit cycles: large delta → positive log_delta
)
```

#### Step 5: Train the Classifier

We provide template initial conditions that we know lead to each attractor type:

```python
knn = KNeighborsClassifier(n_neighbors=1)
classifier = KNNCluster(
    classifier=knn,
    template_y0=[
        [0.0, 0.0],   # This IC leads to the fixed point
        [1.5, 0.0]    # This IC leads to the limit cycle
    ],
    labels=["Fixed Point", "Limit Cycle"],
    ode_params={}
)
```

#### Step 6: Estimate Basin Stability

Now we're ready to run the analysis:

```python
estimator = BasinStabilityEstimator(
    n=1000,                 # Sample 1000 initial conditions
    ode_system=ode_system,
    sampler=sampler,
    solver=solver,
    feature_extractor=feature_extractor,
    estimator=classifier,
    output_dir="my_results"    # Optional: save results to folder
)

# Run the estimation
basin_stability = estimator.estimate_bs()
print(f"Basin stability: {basin_stability}")
# Example output: {'Fixed Point': 0.45, 'Limit Cycle': 0.55}
# This means 45% of initial conditions lead to the fixed point,
# and 55% lead to the limit cycle.
```

### Next Steps

For complete working examples with visualization:

- **Pendulum**: `case_studies/pendulum/` - Forced pendulum with multiple attractors
- **Lorenz System**: `case_studies/lorenz/` - Chaotic attractor analysis
- **Duffing Oscillator**: `case_studies/duffing_oscillator/` - Classic bistable system
- **Friction System**: `case_studies/friction/` - Mechanical system with friction
- **Rössler Network**: `case_studies/rossler_network/` - Coupled oscillators with synchronization

Run a case study:

```bash
uv run python -m case_studies.pendulum.main_pendulum_case1
```

## 📚 Documentation

Full documentation is available at: **[https://adrianwix.github.io/pyBasin/](https://adrianwix.github.io/pyBasin/)**

Or build locally:

```bash
uv add -e ".[docs]"
mkdocs serve
```

Then visit http://localhost:8000

## 🧪 Case Studies

This repository includes validated case studies from the original bSTAB paper and the seminal work by Menck et al. (2013) on basin stability:

| Case Study             | Location                           | Description                         |
| ---------------------- | ---------------------------------- | ----------------------------------- |
| **Duffing Oscillator** | `case_studies/duffing_oscillator/` | Forced oscillator with bistability  |
| **Lorenz System**      | `case_studies/lorenz/`             | Classic chaotic attractor           |
| **Pendulum**           | `case_studies/pendulum/`           | Forced pendulum with bifurcations   |
| **Friction System**    | `case_studies/friction/`           | Mechanical system with friction     |
| **Rössler Network**    | `case_studies/rossler_network/`    | Coupled oscillators synchronization |

Run a case study:

```bash
uv run python -m case_studies.pendulum.main_pendulum_case1
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
│   ├── unit/             # Unit tests for individual components
│   └── integration/      # Validation against MATLAB
├── docs/                 # 📖 Documentation source
├── artifacts/            # 📊 Generated figures and results
├── benchmarks/           # ⚡ Performance benchmarks
├── experiments/          # 🧪 Experiments with different libraries and algorithms
└── scripts/              # 🛠️ Helper scripts
```

## 🧑‍💻 Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/adrianwix/pyBasin.git
cd pyBasinWorkspace

# Install all dependencies including dev tools (creates .venv automatically)
uv sync --all-groups

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# To add a new dependency
uv add <package>

# To add a new dev dependency
uv add --dev <package>
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/pybasin

# Run only integration tests
uv run pytest tests/integration/

# Run only unit tests
uv run pytest tests/unit/

# Run specific case study test
uv run pytest tests/integration/test_duffing.py
```

### Generating Artifacts

Generate documentation artifacts (JSON and plots) after tests pass:

```bash
uv run pytest --generate-artifacts
```

### Code Quality

Run all code quality checks (linter, formatter, and type checker):

```bash
sh scripts/ci.sh
```

## 📊 Validation

pyBasin is validated against the original MATLAB bSTAB implementation. Integration tests in `tests/integration/` compare basin stability values for all case studies.

## 🎓 Academic Context

This library is the main contribution of the bachelor thesis **"Pybasin: A Python Toolbox for Basin Stability of Multi-Stable Dynamical Systems"**. It ports and extends the functionality of the MATLAB bSTAB library with modern Python practices and additional features.

## 🔗 Related Projects

- **bSTAB (MATLAB)**: Original implementation - [GitHub](https://github.com/original/bSTAB)

## 📝 Citation

If you use pyBasin in your research, please cite:

```bibtex
@software{pybasin2025,
  author = {Wix, Adrian},
  title = {Pybasin: A Python Toolbox for Basin Stability of Multi-Stable Dynamical Systems},
  year = {2025},
  url = {https://github.com/adrianwix/pyBasin}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please see the [Contributing Guide](docs/development/contributing.md) for details.

## 📧 Contact

- **Author**: Adrian Wix
- **Repository**: [https://github.com/adrianwix/pyBasin](https://github.com/adrianwix/pyBasin)
- **Issues**: [https://github.com/adrianwix/pyBasin/issues](https://github.com/adrianwix/pyBasin/issues)

---

**Note**: This is an active development project. The API may change before the 1.0 release.
