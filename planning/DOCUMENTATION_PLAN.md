# pyBasin Documentation Plan

> **Epic**: Complete documentation for the pyBasin library for basin stability estimation.
> **Location**: `pyBasinWorkspace/docs/`
> **Build System**: MkDocs with Material theme
> **Target Audience**: Researchers in dynamical systems, engineers, and developers

---

## Table of Contents

1. [Implementation Checklist](#1-implementation-checklist)
2. [Overview & Goals](#2-overview--goals)
3. [Navigation Structure](#3-navigation-structure)
4. [Page Specifications](#4-page-specifications)
5. [Appendix: Source File References](#5-appendix-source-file-references)
6. [Notes for Implementers](#6-notes-for-implementers)

---

## 1. Implementation Checklist

### 1.1 Phase 1: Infrastructure (Priority: High)

| Task                                  | Depends On | Complexity | Status |
| ------------------------------------- | ---------- | ---------- | ------ |
| Update `mkdocs.yml` navigation        | -          | Low        | ✅     |
| Create directory structure in `docs/` | mkdocs.yml | Low        | ✅     |
| Configure mkdocstrings for API docs   | mkdocs.yml | Medium     | ✅     |
| Create `docs/assets/` folder          | -          | Low        | ✅     |

### 1.2 Phase 2: User Guide (Priority: High)

| Task                    | Depends On   | Complexity | Status |
| ----------------------- | ------------ | ---------- | ------ |
| BSE Overview page       | -            | Medium     | ⬜     |
| Parameter Studies page  | BSE Overview | Medium     | ⬜     |
| Samplers page           | -            | Low        | ✅     |
| Solvers page            | -            | Medium     | ⬜     |
| Feature Extractors page | -            | Medium     | ⬜     |
| Feature Selectors page  | -            | Low        | ⬜     |
| Predictors page         | -            | Medium     | ⬜     |
| Plotters page           | Screenshots  | Medium     | ⬜     |

### 1.3 Phase 3: Case Studies (Priority: High)

| Task                       | Depends On       | Complexity | Status |
| -------------------------- | ---------------- | ---------- | ------ |
| Pendulum case study        | Artifact script  | Medium     | ✅     |
| Duffing case study         | Artifact script  | Medium     | ✅     |
| Lorenz case study          | Artifact script  | Medium     | ✅     |
| Friction case study        | Artifact script  | Medium     | ✅     |
| Rössler Network case study | Integration test | High       | ✅     |

### 1.4 Phase 4: Benchmarks (Priority: Medium)

| Task                    | Depends On           | Complexity | Status |
| ----------------------- | -------------------- | ---------- | ------ |
| Benchmarks overview     | -                    | Low        | ✅     |
| Solver comparison page  | Existing data        | Medium     | ✅     |
| Feature extraction page | Existing data        | Medium     | ⬜     |
| End-to-end page         | E2E benchmark script | High       | ✅     |
| Basin stability page    | Existing data        | Medium     | ✅     |

### 1.5 Phase 5: API Reference (Priority: Medium) ✅ COMPLETED

| Task                        | Depends On   | Complexity | Status |
| --------------------------- | ------------ | ---------- | ------ |
| BSE API page                | mkdocstrings | Low        | ✅     |
| AS-BSE API page             | mkdocstrings | Low        | ✅     |
| Solution API page           | mkdocstrings | Low        | ✅     |
| Samplers API page           | mkdocstrings | Low        | ✅     |
| Solvers API page            | mkdocstrings | Low        | ✅     |
| Feature Extractors API page | mkdocstrings | Low        | ✅     |
| Feature Selectors API page  | mkdocstrings | Low        | ✅     |
| Predictors API page         | mkdocstrings | Low        | ✅     |
| Plotters API page           | mkdocstrings | Low        | ✅     |

### 1.6 Phase 6: Artifact Generation (Priority: High - Blocking)

| Task                                    | Depends On | Complexity | Status |
| --------------------------------------- | ---------- | ---------- | ------ |
| Create `generate_docs_artifacts.py`     | -          | High       | ✅     |
| Create Rössler integration test         | -          | Medium     | ✅     |
| Create end-to-end benchmark script      | -          | High       | ✅     |
| Generate MatplotlibPlotter screenshots  | -          | Low        | ✅     |
| Generate InteractivePlotter screenshots | -          | Medium     | ⬜     |
| Run all benchmarks and save results     | Scripts    | Medium     | ✅     |

### 1.7 Phase 7: Polish (Priority: Low)

| Task                                    | Depends On   | Complexity | Status |
| --------------------------------------- | ------------ | ---------- | ------ |
| Update `index.md` home page             | All sections | Low        | ⬜     |
| Update `quickstart.md` with current API | User Guide   | Low        | ⬜     |
| Review and update `installation.md`     | -            | Low        | ⬜     |
| Cross-link between pages                | All pages    | Low        | ⬜     |
| Proofread all content                   | All pages    | Medium     | ⬜     |

---

## 2. Overview & Goals

### 2.1 Purpose

Create comprehensive documentation for pyBasin that enables users to:

- Understand basin stability concepts and the library's approach
- Get started quickly with minimal setup examples
- Customize every component (samplers, solvers, feature extractors, predictors)
- Reproduce case studies from published research
- Understand performance characteristics and scaling behavior

### 2.2 Documentation Principles

- **Minimal examples first**: Every concept starts with the simplest working code
- **Progressive disclosure**: Basic usage → customization → advanced topics
- **Validated examples**: All code examples must match integration test expectations
- **Visual documentation**: Include plots and figures for every case study

### 2.3 Current State

The existing `docs/` folder contains:

| File                                 | Status                                 |
| ------------------------------------ | -------------------------------------- |
| `index.md`                           | ✅ Exists (144 lines) - needs updating |
| `getting-started/installation.md`    | ✅ Exists - needs review               |
| `getting-started/quickstart.md`      | ✅ Exists - uses outdated API          |
| `guides/unbounded-trajectories.md`   | ✅ Exists - comprehensive              |
| `guides/torchode-solver.md`          | ✅ Exists                              |
| `guides/type-safety-generics.md`     | ✅ Exists                              |
| `guides/custom-feature-extractor.md` | ✅ Exists                              |
| `case-studies/overview.md`           | ✅ Exists - incomplete                 |
| All other referenced pages           | ❌ Missing                             |

### 2.4 Technology Stack

- **MkDocs** with **Material** theme
- **mkdocstrings** for API reference auto-generation
- **mkdocs-jupyter** for notebook integration (if needed)
- **pymdownx** extensions for code highlighting, tabs, admonitions

---

## 3. Navigation Structure

### 3.1 Proposed Navigation Hierarchy

```
Home
├── Getting Started
│   ├── Installation
│   └── Quick Start
├── User Guide
│   ├── Basin Stability Estimator
│   │   ├── Overview & Flow
│   │   └── Parameter Studies
│   ├── Samplers
│   ├── Solvers
│   ├── Feature Extractors
│   ├── Feature Selectors
│   ├── Predictors
│   └── Plotters
├── Case Studies
│   ├── Overview
│   ├── Pendulum
│   ├── Duffing Oscillator
│   ├── Lorenz System
│   ├── Friction Oscillator
│   └── Rössler Network
├── Benchmarks
│   ├── Overview
│   ├── End-to-End Performance
│   ├── Solver Comparison
│   └── Feature Extraction
└── API Reference
    ├── BasinStabilityEstimator
    ├── BasinStabilityStudy
    ├── Samplers
    ├── Solvers
    ├── Feature Extractors
    ├── Predictors
    └── Plotters
```

### 3.2 mkdocs.yml Changes Required

Replace the `nav:` section in `mkdocs.yml` (lines 70-97) with:

```yaml
nav:
  - Home: index.md
  - Getting Started:
      - Installation: getting-started/installation.md
      - Quick Start: getting-started/quickstart.md
  - User Guide:
      - Basin Stability Estimator:
          - Overview & Flow: user-guide/bse-overview.md
          - Parameter Studies: user-guide/parameter-studies.md
      - Samplers: user-guide/samplers.md
      - Solvers: user-guide/solvers.md
      - Feature Extractors: user-guide/feature-extractors.md
      - Feature Selectors: user-guide/feature-selectors.md
      - Predictors: user-guide/predictors.md
      - Plotters: user-guide/plotters.md
  - Case Studies:
      - Overview: case-studies/overview.md
      - Pendulum: case-studies/pendulum.md
      - Duffing Oscillator: case-studies/duffing.md
      - Lorenz System: case-studies/lorenz.md
      - Friction Oscillator: case-studies/friction.md
      - Rössler Network: case-studies/rossler-network.md
  - Benchmarks:
      - Overview: benchmarks/overview.md
      - End-to-End Performance: benchmarks/end-to-end.md
      - Solver Comparison: benchmarks/solvers.md
      - Feature Extraction: benchmarks/feature-extraction.md
  - API Reference:
      - BasinStabilityEstimator: api/basin-stability-estimator.md
      - BasinStabilityStudy: api/adaptive-sampling.md
      - Samplers: api/samplers.md
      - Solvers: api/solvers.md
      - Feature Extractors: api/feature-extractors.md
      - Predictors: api/predictors.md
      - Plotters: api/plotters.md
```

Key changes from current `mkdocs.yml`:

1. **Remove** `Theory` section (merge into User Guide where relevant)
2. **Add** `User Guide` section with component-specific pages
3. **Add** `Benchmarks` section with 4 pages
4. **Expand** `API Reference` to cover all public classes
5. **Add** Rössler Network to Case Studies
6. **Remove** `Development` section (not priority for users)

---

## 4. Page Specifications

### 4.1 Home (`index.md`)

**Purpose**: Landing page introducing pyBasin

**Content Outline**:

1. What is pyBasin? (2-3 sentences)
2. Key Features (bullet list)
   - GPU-accelerated ODE integration via JAX/Diffrax
   - Multiple solver backends (JAX, PyTorch, SciPy)
   - Automated feature extraction (~700 time-series features)
   - Unsupervised and supervised classification
   - Interactive visualization dashboard
3. Quick example (10-15 lines showing minimal usage)
4. Installation command (`uv add pybasin` or `pip install pybasin`)
5. Links to Getting Started, Case Studies

**Source Reference**: Use `case_studies/pendulum/main_pendulum_with_defaults.py` pattern

---

### 4.2 Getting Started

#### 4.2.1 Installation (`getting-started/installation.md`)

**Purpose**: Complete installation instructions

**Content Outline**:

1. Requirements (Python 3.12+, CUDA optional)
2. Install via pip/uv
3. Install with GPU support (JAX CUDA)
4. Verify installation (test import)
5. Optional dependencies (tsfresh, nolds)

#### 4.2.2 Quick Start (`getting-started/quickstart.md`)

**Purpose**: First working example in under 5 minutes

**Content Outline**:

1. Define your ODE system (show both PyTorch and JAX patterns)
2. Create a sampler
3. Run `BasinStabilityEstimator`
4. Interpret results
5. Visualize with `MatplotlibPlotter`

**Code Example Pattern** (from `main_pendulum_with_defaults.py`):

```python
from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator

props = setup_pendulum_system()
bse = BasinStabilityEstimator(
    ode_system=props["ode_system"],
    sampler=props["sampler"],
)
basin_stability = bse.estimate_bs()
print(basin_stability)  # {'FP': 0.52, 'LC': 0.48}
```

---

### 4.3 User Guide

#### 4.3.1 Basin Stability Estimator Overview (`user-guide/bse-overview.md`)

**Purpose**: Explain the core estimation workflow and defaults

**Content Outline**:

1. **What is Basin Stability?**
   - Brief theory (link to papers)
   - Monte Carlo sampling approach

2. **The `BasinStabilityEstimator` Class**
   - Constructor parameters table:

     | Parameter           | Type                | Default                  | Description                 |
     | ------------------- | ------------------- | ------------------------ | --------------------------- |
     | `ode_system`        | `ODESystemProtocol` | Required                 | The dynamical system        |
     | `sampler`           | `Sampler`           | Required                 | Initial condition generator |
     | `n`                 | `int`               | `10_000`                 | Number of samples           |
     | `solver`            | `SolverProtocol`    | Auto-detect              | ODE integrator              |
     | `feature_extractor` | `FeatureExtractor`  | `TorchFeatureExtractor`  | Feature computation         |
     | `predictor`         | `BaseEstimator`     | `HDBSCANClusterer`       | Classification method       |
     | `feature_selector`  | `BaseEstimator`     | `DefaultFeatureSelector` | Feature filtering           |
     | `detect_unbounded`  | `bool`              | `True`                   | Stop diverging trajectories |
     | `save_to`           | `str`               | `None`                   | Output directory            |

3. **Default Flow Diagram**

   ```
   Sample ICs → Integrate ODEs → Detect Unbounded → Extract Features
   → Filter Features → Cluster/Classify → Compute BS Values
   ```

4. **Automatic Solver Selection**
   - If `ode_system` is `JaxODESystem` → uses `JaxSolver`
   - If `ode_system` is `ODESystem` → uses `TorchDiffEqSolver`

5. **Unboundedness Detection**
   - Only active when `detect_unbounded=True` AND solver is `JaxSolver` with `event_fn`
   - Separates unbounded trajectories before feature extraction
   - Labels them as "unbounded" in final results
   - Reference: `docs/guides/unbounded-trajectories.md`

6. **Feature Filtering**
   - Default: `DefaultFeatureSelector` removes constant/low-variance features
   - Can pass any sklearn transformer (`VarianceThreshold`, `SelectKBest`, etc.)
   - Pass `feature_selector=None` to disable

7. **Output Attributes**
   - `bse.bs_vals`: Dict of basin stability values per class
   - `bse.y0`: Initial conditions tensor
   - `bse.solution`: Solution object with trajectories, features, labels

**Source Files**:

- `src/pybasin/basin_stability_estimator.py` (694 lines)
- `src/pybasin/protocols.py` for type definitions

#### 4.3.2 Parameter Studies (`user-guide/parameter-studies.md`)

**Purpose**: Explain `BasinStabilityStudy` for parameter sweeps

**Content Outline**:

1. **Use Case**: Study how basin stability changes with a system parameter

2. **The `BasinStabilityStudy` Class**
   - Runs BSE multiple times for different parameter configurations
   - Returns parameter values, BS values, and full results per run

3. **Example: Pendulum Damping Study**

   ```python
   import numpy as np
   from pybasin.basin_stability_study import BasinStabilityStudy
   from pybasin.study_params import SweepStudyParams

   study_params = SweepStudyParams(
       name='ode_system.params["gamma"]',
       values=np.linspace(0.1, 0.5, 10),
   )

   study = BasinStabilityStudy(
       n=10_000,
       ode_system=pendulum_ode,
       sampler=sampler,
       solver=solver,
       feature_extractor=feature_extractor,
       estimator=predictor,
       study_params=study_params,
   )
   labels, bs_vals, results = study.estimate_as_bs()
   ```

**Source Files**:

- `src/pybasin/basin_stability_study.py` (264 lines)

#### 4.3.3 Samplers (`user-guide/samplers.md`)

Implemented

#### 4.3.4 Solvers (`user-guide/solvers.md`)

**Purpose**: Document all ODE solver backends

**Content Outline**:

1. **Solver Protocol**
   - Method: `integrate(ode_system, initial_conditions) -> (t, y)`
   - Property: `device`

2. **Available Solvers**:

   | Class                   | Backend       | GPU Support | Event Functions     | Recommended For             |
   | ----------------------- | ------------- | ----------- | ------------------- | --------------------------- |
   | `JaxSolver`             | JAX/Diffrax   | ✅ CUDA     | ✅ Yes              | **Default for performance** |
   | `TorchDiffEqSolver`     | torchdiffeq   | ✅ CUDA     | ❌ Batch limitation | PyTorch ecosystems          |
   | `TorchOdeSolver`        | torchode      | ✅ CUDA     | ❌ No               | Alternative PyTorch         |
   | `SklearnParallelSolver` | scipy/sklearn | ❌ CPU only | ❌ No               | Debugging, reference        |

3. **JaxSolver (Recommended)**

   ```python
   from pybasin.solvers.jax_solver import JaxSolver

   solver = JaxSolver(
       device="cuda",
       t_span=(0.0, 1000.0),
       n_steps=1000,
       method="Dopri5",  # or "Tsit5"
       rtol=1e-8,
       atol=1e-6,
       event_fn=my_stop_event,  # Optional: stop unbounded
   )
   ```

4. **Event Functions for Unbounded Detection**

   ```python
   import jax.numpy as jnp

   def stop_event(t, y, args):
       """Stop when |y| > 200."""
       return 200.0 - jnp.max(jnp.abs(y))
   ```

5. **TorchDiffEqSolver**

   ```python
   from pybasin.solver import TorchDiffEqSolver

   solver = TorchDiffEqSolver(
       device="cuda",
       t_span=(0.0, 1000.0),
       method="dopri5",
   )
   ```

6. **Performance Comparison** (link to Benchmarks section)

**Source Files**:

- `src/pybasin/solvers/jax_solver.py` (301 lines)
- `src/pybasin/solver.py` (597 lines)

#### 4.3.5 Feature Extractors (`user-guide/feature-extractors.md`)

**Purpose**: Document feature extraction from trajectories

**Content Outline**:

1. **What are Features?**
   - Time-series characteristics (mean, variance, entropy, etc.)
   - Used to distinguish different attractor types

2. **Base Class**: `FeatureExtractor`
   - Method: `extract_features(solution: Solution) -> torch.Tensor`
   - Property: `feature_names`

3. **Available Extractors**:

   | Class                      | Features | GPU | Speed   | Use Case                |
   | -------------------------- | -------- | --- | ------- | ----------------------- |
   | `TorchFeatureExtractor`    | ~700     | ✅  | Fast    | **Default**             |
   | `JaxFeatureExtractor`      | ~50      | ✅  | Fastest | JAX-only workflows      |
   | `TsFreshFeatureExtractor`  | ~700     | ❌  | Slow    | Reference/validation    |
   | `NoldsFeatureExtractor`    | ~10      | ❌  | Slow    | Dynamical features only |
   | `StateSpaceStatsExtractor` | ~20      | ❌  | Fast    | Simple statistics       |

4. **TorchFeatureExtractor (Default)**

   ```python
   from pybasin.feature_extractors import TorchFeatureExtractor

   extractor = TorchFeatureExtractor(
       fc_parameters="minimal",  # or "comprehensive", custom dict
       time_steady=800.0,  # Discard transient
       device="cuda",
   )
   ```

   Feature categories:
   - Statistical (mean, variance, skewness, kurtosis)
   - Frequency (FFT coefficients, spectral entropy)
   - Autocorrelation
   - Change detection (change quantiles)
   - Entropy measures
   - Dynamical (Lyapunov exponents, correlation dimension)

5. **Creating Custom Feature Extractors**

   Example from Rössler Network synchronization study:

   ```python
   # From case_studies/rossler_network/synchronization_feature_extractor.py

   class SynchronizationFeatureExtractor(FeatureExtractor):
       def __init__(self, n_nodes: int, time_steady: float = 1000.0):
           super().__init__(time_steady=time_steady)
           self.n_nodes = n_nodes
           self._feature_names = [
               "max_deviation_x", "max_deviation_y",
               "max_deviation_z", "max_deviation_all"
           ]

       def extract_features(self, solution: Solution) -> torch.Tensor:
           y_filtered = self.filter_time(solution)
           # Compute max |x_i - x_j| across all node pairs
           ...
           return features
   ```

6. **Combining Multiple Extractors**
   - Use `CompositeFeatureExtractor` to stack features

**Source Files**:

- `src/pybasin/feature_extractors/feature_extractor.py` (125 lines)
- `src/pybasin/feature_extractors/jax_feature_extractor.py`
- `src/pybasin/ts_torch/torch_feature_extractor.py` (299 lines)
- `case_studies/rossler_network/synchronization_feature_extractor.py`

#### 4.3.6 Feature Selectors (`user-guide/feature-selectors.md`)

**Purpose**: Document feature filtering before classification

**Content Outline**:

1. **Why Filter Features?**
   - Remove constant/low-variance features
   - Reduce dimensionality for clustering
   - Remove correlated features

2. **Default Behavior**
   - `DefaultFeatureSelector` removes features with zero variance

3. **Using sklearn Transformers**

   ```python
   from sklearn.feature_selection import VarianceThreshold

   bse = BasinStabilityEstimator(
       ...,
       feature_selector=VarianceThreshold(threshold=0.01),
   )
   ```

4. **CorrelationSelector**

   ```python
   from pybasin.feature_extractors.correlation_selector import CorrelationSelector

   selector = CorrelationSelector(threshold=0.95)
   ```

5. **Disabling Feature Selection**
   ```python
   bse = BasinStabilityEstimator(..., feature_selector=None)
   ```

**Source Files**:

- `src/pybasin/feature_extractors/default_feature_selector.py`
- `src/pybasin/feature_extractors/correlation_selector.py`

#### 4.3.7 Predictors (`user-guide/predictors.md`)

**Purpose**: Document classification/clustering methods

**Content Outline**:

1. **Predictor Types**:
   - **Unsupervised** (sklearn clusterers with `fit_predict()`): Discovers attractor classes automatically
   - **Supervised** (sklearn classifiers with `fit()` + `predict()`): Uses known template trajectories

2. **Available Predictors**:

   | Class                        | Type         | Description                                           |
   | ---------------------------- | ------------ | ----------------------------------------------------- |
   | `HDBSCANClusterer`           | Unsupervised | **Default**, density-based, auto-tunes parameters     |
   | `DBSCANClusterer`            | Unsupervised | DBSCAN with automatic epsilon tuning                  |
   | `DynamicalSystemClusterer`   | Unsupervised | Physics-based two-stage clustering                    |
   | `UnboundednessMetaEstimator` | Meta         | Wraps any estimator, separates unbounded trajectories |
   | Any sklearn classifier       | Supervised   | e.g. `KNeighborsClassifier` with templates            |

3. **HDBSCANClusterer (Default)**

   ```python
   from sklearn.cluster import HDBSCAN
   from pybasin.predictors.hdbscan_clusterer import HDBSCANClusterer

   predictor = HDBSCANClusterer(
       hdbscan=HDBSCAN(min_cluster_size=50, min_samples=10, copy=True),
       auto_tune=True,      # Auto-select min_cluster_size
       assign_noise=True,   # Assign noise points to nearest cluster
   )
   ```

4. **DBSCANClusterer (with auto-tuning)**

   ```python
   from sklearn.cluster import DBSCAN
   from pybasin.predictors.dbscan_clusterer import DBSCANClusterer

   predictor = DBSCANClusterer(
       dbscan=DBSCAN(eps=0.5, min_samples=10),
       auto_tune=True,       # Automatic epsilon search via silhouette analysis
       assign_noise=False,
   )
   ```

5. **Supervised Classification (sklearn classifiers)**

   ```python
   from sklearn.neighbors import KNeighborsClassifier

   predictor = KNeighborsClassifier(n_neighbors=5)
   # Requires template_integrator in BasinStabilityEstimator
   ```

6. **Creating Custom Predictors**

   Any sklearn-compatible estimator works. Clusterers need `fit_predict()`,
   classifiers need `fit()` + `predict()`:

   ```python
   # From case_studies/rossler_network/synchronization_classifier.py

   class SynchronizationClassifier(BaseEstimator, ClusterMixin):
       def __init__(self, epsilon: float = 0.1):
           self.epsilon = epsilon

       def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
           max_deviation = X[:, 3]  # max_deviation_all
           return np.where(
               max_deviation < self.epsilon,
               "synchronized",
               "desynchronized",
           )
   ```

**Source Files**:

- `src/pybasin/predictors/hdbscan_clusterer.py`
- `src/pybasin/predictors/dbscan_clusterer.py`
- `src/pybasin/predictors/dynamical_system_clusterer.py`
- `src/pybasin/predictors/unboundedness_meta_estimator.py`
- `case_studies/rossler_network/synchronization_classifier.py`

#### 4.3.8 Plotters (`user-guide/plotters.md`)

**Purpose**: Document visualization options

**Content Outline**:

1. **Available Plotters**:

   | Class                | Type    | Use Case                              |
   | -------------------- | ------- | ------------------------------------- |
   | `MatplotlibPlotter`  | Static  | Publication figures, quick inspection |
   | `InteractivePlotter` | Web app | Exploration, presentations            |

2. **MatplotlibPlotter**

   ```python
   from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter

   plotter = MatplotlibPlotter(bse)
   plotter.plot_bse_results()      # 4-panel diagnostic plot
   plotter.plot_phase(x_var=0, y_var=1)  # Phase space
   plotter.plot_templates(plotted_var=0) # Template time series
   ```

   Methods:
   - `plot_bse_results()`: 4-panel plot (BS bar chart, state space, feature space, placeholder)
   - `plot_phase(x_var, y_var, z_var=None)`: 2D/3D phase plot of templates
   - `plot_templates(plotted_var, time_span=None)`: Time series of templates

3. **InteractivePlotter**

   ```python
   from pybasin.plotters.interactive_plotter import InteractivePlotter

   plotter = InteractivePlotter(
       bse,
       state_labels={0: "θ", 1: "ω"},
   )
   plotter.run(port=8050)  # Opens web browser
   ```

   Pages (tabs):
   - **State Space**: Scatter plot of ICs colored by label
   - **Feature Space**: 2D/3D scatter of features
   - **Basin Stability**: Bar chart
   - **Template Phase Plot**: Phase space trajectories
   - **Template Time Series**: State vs time
   - **Trajectory Modal**: Click to inspect individual trajectories

4. **Screenshots to Include**
   - MatplotlibPlotter `plot_bse_results()` 4-panel output
   - InteractivePlotter State Space page
   - InteractivePlotter Feature Space page
   - InteractivePlotter Basin Stability bar chart

**Source Files**:

- `src/pybasin/plotters/matplotlib_plotter.py` (240 lines)
- `src/pybasin/plotters/interactive_plotter/plotter.py` (662 lines)

---

### 4.4 API Reference

All API reference pages use `mkdocstrings` for auto-generation from docstrings.

Each page follows this pattern:

```markdown
# ClassName

::: pybasin.module.ClassName
options:
show_source: true
members_order: source
```

#### Pages to Create:

| Page                               | Module Path                                                 |
| ---------------------------------- | ----------------------------------------------------------- |
| `api/basin-stability-estimator.md` | `pybasin.basin_stability_estimator.BasinStabilityEstimator` |
| `api/adaptive-sampling.md`         | `pybasin.basin_stability_study.BasinStabilityStudy`         |
| `api/samplers.md`                  | `pybasin.sampler` (all classes)                             |
| `api/solvers.md`                   | `pybasin.solver`, `pybasin.solvers.jax_solver`              |
| `api/feature-extractors.md`        | `pybasin.feature_extractors`                                |
| `api/predictors.md`                | `pybasin.predictors`                                        |
| `api/plotters.md`                  | `pybasin.plotters`                                          |

---

## 5. Appendix: Source File References

### 5.1 Core Library (`src/pybasin/`)

| File                           | Lines | Key Classes                                                    |
| ------------------------------ | ----- | -------------------------------------------------------------- |
| `basin_stability_estimator.py` | 694   | `BasinStabilityEstimator`                                      |
| `basin_stability_study.py`     | 264   | `BasinStabilityStudy`                                          |
| `sampler.py`                   | ~150  | `Sampler`, `UniformRandomSampler`, `GridSampler`               |
| `solver.py`                    | 597   | `TorchDiffEqSolver`, `TorchOdeSolver`, `SklearnParallelSolver` |
| `solvers/jax_solver.py`        | 301   | `JaxSolver`                                                    |
| `ode_system.py`                | ~100  | `ODESystem`                                                    |
| `jax_ode_system.py`            | 144   | `JaxODESystem`                                                 |
| `solution.py`                  | ~150  | `Solution`                                                     |
| `protocols.py`                 | ~50   | `ODESystemProtocol`, `SolverProtocol`                          |

### 5.2 Feature Extractors (`src/pybasin/feature_extractors/`)

| File                           | Key Classes               |
| ------------------------------ | ------------------------- |
| `feature_extractor.py`         | `FeatureExtractor` (base) |
| `jax_feature_extractor.py`     | `JaxFeatureExtractor`     |
| `tsfresh_feature_extractor.py` | `TsfreshFeatureExtractor` |
| `default_feature_selector.py`  | `DefaultFeatureSelector`  |
| `correlation_selector.py`      | `CorrelationSelector`     |

### 5.3 Predictors (`src/pybasin/predictors/`)

| File                              | Key Classes                                                |
| --------------------------------- | ---------------------------------------------------------- |
| `hdbscan_clusterer.py`            | `HDBSCANClusterer`                                         |
| `dbscan_clusterer.py`             | `DBSCANClusterer`                                          |
| `dynamical_system_clusterer.py`   | `DynamicalSystemClusterer`                                 |
| `unboundedness_meta_estimator.py` | `UnboundednessMetaEstimator`, `default_unbounded_detector` |

### 5.4 Plotters (`src/pybasin/plotters/`)

| File                             | Key Classes          |
| -------------------------------- | -------------------- |
| `matplotlib_plotter.py`          | `MatplotlibPlotter`  |
| `interactive_plotter/plotter.py` | `InteractivePlotter` |

### 5.5 Case Studies (`case_studies/`)

| System   | Directory             | Key Files                                                             |
| -------- | --------------------- | --------------------------------------------------------------------- |
| Pendulum | `pendulum/`           | `main_pendulum_with_defaults.py`, `setup_pendulum_system.py`          |
| Duffing  | `duffing_oscillator/` | `main_duffing_oscillator_with_defaults.py`, `setup_duffing_system.py` |
| Lorenz   | `lorenz/`             | `main_lorenz_with_defaults.py`, `setup_lorenz_system.py`              |
| Friction | `friction/`           | `main_friction_with_defaults.py`, `setup_friction_system.py`          |
| Rössler  | `rossler_network/`    | `main_rossler_network.py`, `setup_rossler_network.py`                 |

### 5.6 Integration Tests (`tests/integration/`)

| System   | Directory   | Expected Results                                                 |
| -------- | ----------- | ---------------------------------------------------------------- |
| Pendulum | `pendulum/` | `main_pendulum_case1.json`, `main_pendulum_case2.json`           |
| Duffing  | `duffing/`  | `main_duffing_supervised.json`, `main_duffing_unsupervised.json` |
| Lorenz   | `lorenz/`   | `main_lorenz.json`, `main_lorenz_sigma_study.json`               |
| Friction | `friction/` | `main_friction_case1.json`, `main_friction_v_study.json`         |
| Rössler  | ❌ Missing  | ❌ TODO: Create                                                  |

### 5.7 Benchmarks (`benchmarks/`)

| Category           | Directory/File                                       | Status          |
| ------------------ | ---------------------------------------------------- | --------------- |
| Solver timing      | `time_integrations/results/`                         | ✅ Complete     |
| Feature extraction | `feature_extraction/`, `batch_benchmark_results.csv` | ✅ Partial      |
| End-to-end         | ❌ Missing                                           | ❌ TODO: Create |

---

## 6. Notes for Implementers

### 6.1 Code Example Guidelines

- All code examples must be runnable
- Use `# Output:` comments to show expected output
- Keep examples under 30 lines when possible
- Always show imports

### 6.2 Figure Guidelines

- Save all figures as PNG at 300 DPI
- Use consistent color scheme across case studies
- Include axis labels and legends
- Maximum width: 800px for documentation

### 6.3 Cross-References

Use MkDocs relative links:

```markdown
See [Solvers](../user-guide/solvers.md) for details.
See the [JaxSolver API](../api/solvers.md#pybasin.solvers.jax_solver.JaxSolver).
```

### 6.4 Admonitions

Use Material theme admonitions for tips, warnings, notes:

```markdown
!!! tip "Performance Tip"
Use JaxSolver with CUDA for sample sizes > 50,000.

!!! warning "Breaking Change"
The `estimator` parameter was renamed to `predictor` in v2.0.

!!! note
This feature requires `detect_unbounded=True` and a JaxSolver with `event_fn`.
```
