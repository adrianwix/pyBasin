# Multi-Parameter Study API Proposal

## Problem Statement

The current `ASBasinStabilityEstimator` only supports varying a single parameter at a time. Real-world basin stability studies often require:

1. **2D Parameter Grids**: Studying basin stability as a function of two parameters (e.g., coupling strength K vs. rewiring probability p)
2. **Mixed Parameter Types**: Varying both ODE parameters and hyperparameters (N samples, solver tolerance, network topology)
3. **Non-Cartesian Studies**: Parameter combinations that aren't simple grids (e.g., different network instances per parameter point)

### Current Limitations

```python
# Current API - single parameter only
as_params = AdaptiveStudyParams(
    adaptative_parameter_values=np.arange(0.01, 0.97, 0.05),
    adaptative_parameter_name='ode_system.params["T"]',
)
```

### Motivating Examples

1. **Pendulum T-study** (`main_pendulum_case2.py`): 1D sweep over period T
2. **Rössler Network 2D** (`main_rossler_network_2_dimensional.py`): 2D grid over (K, p) - currently implemented with manual nested loops
3. **Hyperparameter sensitivity**: N samples vs. solver tolerance grid

### Key Discovery: Current eval() Pattern Already Supports Object Parameters

The current `ASBasinStabilityEstimator` implementation using `eval()` already works with object parameters like `Sampler` instances:

```python
# This WORKS with current implementation!
samplers = [CsvSampler(f"gt_T_{t}.csv") for t in t_values]
as_params = AdaptiveStudyParams(
    adaptative_parameter_values=np.asarray(samplers, dtype=object),
    adaptative_parameter_name="sampler",
)
```

This means we can extend the API without rewriting the estimator - just provide smarter parameter generators.

---

## ⭐ Proposal 7: Study Params Generators (RECOMMENDED)

Keep the current `ASBasinStabilityEstimator` unchanged. Instead, create generator classes that produce `StudyConfig` objects - a list of parameter assignments per run that the existing `eval()` mechanism can apply.

### Core Idea

- The estimator's `eval()` pattern already works - don't change it
- Create generator classes that produce parameter combinations
- Each generator outputs a `StudyConfig` with a list of `(param_name, value)` tuples per run
- Support single params (current), grids, zips, and custom combinations
- The estimator iterates through configs, applying all assignments per run

### API Design

```python
from dataclasses import dataclass
from typing import Any, Iterator
from itertools import product

@dataclass
class ParamAssignment:
    """A single parameter assignment."""
    name: str  # e.g., 'ode_system.params["T"]', 'sampler', 'n'
    value: Any

@dataclass
class RunConfig:
    """Configuration for a single BSE run - multiple parameter assignments."""
    assignments: list[ParamAssignment]
    label: dict[str, Any]  # For results indexing, e.g., {"T": 0.5, "p": 0.2}


class StudyParams:
    """Base class for study parameter generators."""

    def __iter__(self) -> Iterator[RunConfig]:
        """Yield RunConfig for each parameter combination."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Total number of runs."""
        raise NotImplementedError


class SweepStudyParams(StudyParams):
    """Single parameter sweep (current behavior)."""

    def __init__(self, name: str, values: np.ndarray | list[Any]):
        self.name = name
        self.values = list(values)

    def __iter__(self) -> Iterator[RunConfig]:
        for val in self.values:
            yield RunConfig(
                assignments=[ParamAssignment(self.name, val)],
                label={self._short_name(): val},
            )

    def __len__(self) -> int:
        return len(self.values)

    def _short_name(self) -> str:
        # Extract short name from 'ode_system.params["T"]' -> 'T'
        if '["' in self.name:
            return self.name.split('["')[1].rstrip('"]')
        return self.name.split(".")[-1]


class GridStudyParams(StudyParams):
    """Cartesian product of multiple parameters."""

    def __init__(self, **params: np.ndarray | list[Any]):
        """
        :param params: Keyword arguments mapping param names to value arrays.
                       e.g., GridStudyParams(T=t_values, sigma=sigma_values)
        """
        self.param_names = list(params.keys())
        self.param_values = [list(v) for v in params.values()]

    def __iter__(self) -> Iterator[RunConfig]:
        for combo in product(*self.param_values):
            assignments = [
                ParamAssignment(name, val)
                for name, val in zip(self.param_names, combo)
            ]
            label = dict(zip(self._short_names(), combo))
            yield RunConfig(assignments=assignments, label=label)

    def __len__(self) -> int:
        result = 1
        for vals in self.param_values:
            result *= len(vals)
        return result

    def _short_names(self) -> list[str]:
        return [
            name.split('["')[1].rstrip('"]') if '["' in name else name.split(".")[-1]
            for name in self.param_names
        ]


class ZipStudyParams(StudyParams):
    """Parallel iteration of multiple parameters (must have same length)."""

    def __init__(self, **params: np.ndarray | list[Any]):
        self.param_names = list(params.keys())
        self.param_values = [list(v) for v in params.values()]

        # Validate same length
        lengths = [len(v) for v in self.param_values]
        if len(set(lengths)) > 1:
            raise ValueError(f"All parameter arrays must have same length, got {lengths}")

    def __iter__(self) -> Iterator[RunConfig]:
        for combo in zip(*self.param_values):
            assignments = [
                ParamAssignment(name, val)
                for name, val in zip(self.param_names, combo)
            ]
            label = dict(zip(self._short_names(), combo))
            yield RunConfig(assignments=assignments, label=label)

    def __len__(self) -> int:
        return len(self.param_values[0]) if self.param_values else 0

    def _short_names(self) -> list[str]:
        return [
            name.split('["')[1].rstrip('"]') if '["' in name else name.split(".")[-1]
            for name in self.param_names
        ]


class CustomStudyParams(StudyParams):
    """User-provided list of configurations."""

    def __init__(self, configs: list[RunConfig]):
        self.configs = configs

    def __iter__(self) -> Iterator[RunConfig]:
        yield from self.configs

    def __len__(self) -> int:
        return len(self.configs)

    @classmethod
    def from_dicts(
        cls,
        param_dicts: list[dict[str, Any]],
    ) -> "CustomStudyParams":
        """Create from list of {param_name: value} dicts."""
        configs = [
            RunConfig(
                assignments=[ParamAssignment(k, v) for k, v in d.items()],
                label=d.copy(),
            )
            for d in param_dicts
        ]
        return cls(configs)
```

### Updated ASBasinStabilityEstimator

Minimal changes to support the new `StudyParams`:

```python
class ASBasinStabilityEstimator:
    def __init__(
        self,
        n: int,
        ode_system: ODESystemProtocol,
        sampler: Sampler,
        solver: SolverProtocol,
        feature_extractor: FeatureExtractor,
        cluster_classifier: LabelPredictor,
        study_params: StudyParams,  # NEW: replaces as_params
        save_to: str | None = "results",
        verbose: bool = False,
    ):
        ...

    def estimate_as_bs(self):
        for run_config in self.study_params:
            # Build context with base components
            context: dict[str, Any] = {
                "n": self.n,
                "ode_system": self.ode_system,
                "sampler": self.sampler,
                "solver": self.solver,
                "feature_extractor": self.feature_extractor,
                "cluster_classifier": self.cluster_classifier,
            }

            # Apply all parameter assignments for this run
            for assignment in run_config.assignments:
                context["_param_value"] = assignment.value
                exec_code = f"{assignment.name} = _param_value"
                eval(compile(exec_code, "<string>", "exec"), context, context)

            # Create and run BSE with updated context
            bse = BasinStabilityEstimator(
                n=context["n"],
                ode_system=context["ode_system"],
                sampler=context["sampler"],
                ...
            )
            ...
```

### Usage Examples

#### Simple 1D Sweep (Current Behavior)

```python
study_params = SweepStudyParams(
    name='ode_system.params["T"]',
    values=np.arange(0.01, 0.97, 0.05),
)

bse = ASBasinStabilityEstimator(
    n=1000,
    ode_system=ode,
    sampler=sampler,
    solver=solver,
    feature_extractor=fe,
    cluster_classifier=cc,
    study_params=study_params,
)
```

#### 2D Grid Study

```python
study_params = GridStudyParams(
    **{
        'ode_system.params["K"]': k_values,
        'ode_system.params["sigma"]': sigma_values,
    }
)
# Runs: K[0]×sigma[0], K[0]×sigma[1], ..., K[n]×sigma[m]
```

#### With Custom Samplers (for Testing)

```python
t_values = np.arange(0.01, 0.97, 0.05)
samplers = [CsvSampler(f"ground_truth_T_{t:.2f}.csv") for t in t_values]

# Option 1: Zip T values with their samplers
study_params = ZipStudyParams(
    **{
        'ode_system.params["T"]': t_values,
        'sampler': samplers,
    }
)

# Option 2: Just vary sampler (T is set inside sampler or ODE setup)
study_params = SweepStudyParams(name='sampler', values=samplers)
```

#### Mixed ODE + Hyperparameters

```python
study_params = GridStudyParams(
    **{
        'ode_system.params["T"]': t_values,
        'solver.rtol': [1e-3, 1e-6, 1e-9],
        'n': [100, 500, 1000],
    }
)
```

#### Complex Custom Combinations

```python
# For Rössler network where topology changes with p
configs: list[RunConfig] = []
for K, p in product(k_values, p_values):
    graph = nx.watts_strogatz_graph(n=400, k=8, p=p, seed=42)
    ode = build_rossler_ode(K, graph)

    configs.append(RunConfig(
        assignments=[
            ParamAssignment('ode_system', ode),
            ParamAssignment('ode_system.params["K"]', K),  # For logging
        ],
        label={"K": K, "p": p},
    ))

study_params = CustomStudyParams(configs)
```

### Backward Compatibility

Keep `AdaptiveStudyParams` working:

```python
# Old API still works
as_params = AdaptiveStudyParams(
    adaptative_parameter_values=np.arange(0.01, 0.97, 0.05),
    adaptative_parameter_name='ode_system.params["T"]',
)

# Internally converted to SweepStudyParams
study_params = SweepStudyParams(
    name=as_params["adaptative_parameter_name"],
    values=as_params["adaptative_parameter_values"],
)
```

### Results with Labels

```python
@dataclass
class MultiStudyResults:
    """Results indexed by label."""
    labels: list[dict[str, Any]]  # From RunConfig.label
    basin_stabilities: list[dict[str, float]]
    errors: list[dict[str, ErrorInfo]]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for label, bs in zip(self.labels, self.basin_stabilities):
            row = {**label, **bs}
            rows.append(row)
        return pd.DataFrame(rows)

    def get_by_label(self, **kwargs) -> dict[str, float]:
        for label, bs in zip(self.labels, self.basin_stabilities):
            if all(label.get(k) == v for k, v in kwargs.items()):
                return bs
        raise KeyError(f"No result found for {kwargs}")
```

### Pros

- **Minimal changes to estimator**: Just update the iteration logic
- **Current eval() pattern works**: No need to rewrite parameter application
- **Backward compatible**: `AdaptiveStudyParams` still works
- **Type-safe generators**: Each generator class has clear semantics
- **Flexible**: Grid, Zip, Sweep, Custom all supported
- **Labels included**: Results are labeled for easy retrieval

### Cons

- **String parameter names**: Still uses `'ode_system.params["T"]'` strings
- **eval() remains**: Same security considerations as before

---

## Proposal 5: Configuration Generator Pattern

The simplest and most flexible approach: user provides a generator/list of `BSEConfig` dataclasses, each representing one complete BSE run configuration.

### Core Idea

- No grid/zip modes - user builds their own list of configurations however they want
- Each configuration is a complete, self-contained setup for one BSE run
- The estimator just iterates through configurations
- Maximum flexibility with minimal API surface

### API Design

```python
from dataclasses import dataclass
from typing import Iterator, Any

@dataclass
class BSERunConfig:
    """Complete configuration for a single BasinStabilityEstimator run."""
    # Required
    ode_system: ODESystemProtocol
    sampler: Sampler
    solver: SolverProtocol
    feature_extractor: FeatureExtractor
    cluster_classifier: LabelPredictor
    n: int

    # Metadata for results indexing/labeling
    label: str | dict[str, Any]  # e.g., "T=0.5" or {"T": 0.5, "p": 0.2}


class MultiStudyEstimator:
    """Basin stability estimator that runs multiple configurations."""

    def __init__(
        self,
        configs: list[BSERunConfig] | Iterator[BSERunConfig],
        save_to: str | None = None,
    ):
        self.configs = list(configs)  # Materialize if iterator
        ...

    def estimate(self) -> "MultiStudyResults":
        """Run BSE for each configuration."""
        ...
```

### Usage Examples

#### Simple 1D Parameter Sweep

```python
def make_pendulum_configs(t_values: np.ndarray) -> list[BSERunConfig]:
    """Generate configs for pendulum T-study."""
    configs: list[BSERunConfig] = []
    for t in t_values:
        props = setup_pendulum_system(T=t)  # Your existing setup function
        configs.append(BSERunConfig(
            ode_system=props["ode_system"],
            sampler=props["sampler"],
            solver=props["solver"],
            feature_extractor=props["feature_extractor"],
            cluster_classifier=props["cluster_classifier"],
            n=props["n"],
            label={"T": t},
        ))
    return configs

# Usage
estimator = MultiStudyEstimator(make_pendulum_configs(np.arange(0.01, 0.97, 0.05)))
results = estimator.estimate()
```

#### With Ground Truth CSV Samplers (for Testing)

```python
def make_test_configs(t_values: np.ndarray, csv_dir: Path) -> list[BSERunConfig]:
    """Generate configs with MATLAB ground truth samplers."""
    configs: list[BSERunConfig] = []
    for t in t_values:
        props = setup_pendulum_system(T=t)

        # Use CsvSampler with exact MATLAB initial conditions
        csv_path = csv_dir / f"ground_truth_T_{t:.2f}.csv"
        sampler = CsvSampler(csv_path, coordinate_columns=["x1", "x2"])

        configs.append(BSERunConfig(
            ode_system=props["ode_system"],
            sampler=sampler,  # Different sampler per T!
            solver=props["solver"],
            feature_extractor=props["feature_extractor"],
            cluster_classifier=props["cluster_classifier"],
            n=sampler.n_samples,
            label={"T": t},
        ))
    return configs
```

#### 2D Grid (Cartesian Product)

```python
from itertools import product

def make_rossler_2d_configs(k_values: np.ndarray, p_values: np.ndarray) -> list[BSERunConfig]:
    """Generate configs for 2D K×p study."""
    configs: list[BSERunConfig] = []

    for K, p in product(k_values, p_values):
        # Build network for this p
        graph = nx.watts_strogatz_graph(n=400, k=8, p=p, seed=42)
        edges_i, edges_j = build_edge_arrays(graph)

        ode_system = RosslerNetworkJaxODE({
            "K": K,
            "edges_i": edges_i,
            "edges_j": edges_j,
            ...
        })

        configs.append(BSERunConfig(
            ode_system=ode_system,
            sampler=sampler,
            solver=solver,
            feature_extractor=fe,
            cluster_classifier=cc,
            n=1000,
            label={"K": K, "p": p},
        ))

    return configs

# Usage
estimator = MultiStudyEstimator(make_rossler_2d_configs(k_values, p_values))
results = estimator.estimate()

# Access 2D results
sb_matrix = results.to_2d_array("K", "p", attractor="synchronized")
```

#### Parallel Sweep (Zip)

```python
def make_zip_configs(sigma_values: np.ndarray, rho_values: np.ndarray) -> list[BSERunConfig]:
    """Generate configs where sigma[i] pairs with rho[i]."""
    assert len(sigma_values) == len(rho_values)

    configs: list[BSERunConfig] = []
    for sigma, rho in zip(sigma_values, rho_values):
        props = setup_lorenz_system(sigma=sigma, rho=rho)
        configs.append(BSERunConfig(
            ...
            label={"sigma": sigma, "rho": rho},
        ))
    return configs
```

#### Custom Non-Grid Combinations

```python
def make_custom_configs() -> list[BSERunConfig]:
    """Any arbitrary combinations you want."""
    combinations = [
        {"K": 0.1, "seed": 42, "n": 500},
        {"K": 0.2, "seed": 43, "n": 1000},
        {"K": 0.1, "seed": 44, "n": 500},
        # ... whatever pattern makes sense
    ]

    configs: list[BSERunConfig] = []
    for combo in combinations:
        graph = nx.watts_strogatz_graph(n=400, k=8, p=0.5, seed=combo["seed"])
        ...
        configs.append(BSERunConfig(..., n=combo["n"], label=combo))

    return configs
```

### Results Structure

```python
@dataclass
class MultiStudyResults:
    """Results from multi-configuration study."""
    configs: list[BSERunConfig]  # Original configs
    basin_stabilities: list[dict[str, float]]  # One per config
    errors: list[dict[str, ErrorInfo]]  # One per config

    def get_by_label(self, **kwargs) -> tuple[dict[str, float], dict[str, ErrorInfo]]:
        """Get results for specific label values."""
        for i, cfg in enumerate(self.configs):
            if all(cfg.label.get(k) == v for k, v in kwargs.items()):
                return self.basin_stabilities[i], self.errors[i]
        raise KeyError(f"No config found with {kwargs}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        rows = []
        for i, cfg in enumerate(self.configs):
            row = dict(cfg.label) if isinstance(cfg.label, dict) else {"label": cfg.label}
            row.update(self.basin_stabilities[i])
            rows.append(row)
        return pd.DataFrame(rows)

    def to_2d_array(
        self,
        x_param: str,
        y_param: str,
        attractor: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reshape results to 2D grid for plotting.

        Returns (x_values, y_values, sb_matrix) for heatmap.
        """
        df = self.to_dataframe()
        pivot = df.pivot(index=y_param, columns=x_param, values=attractor)
        return pivot.columns.values, pivot.index.values, pivot.values
```

### Pros

- **Maximum flexibility**: Any combination pattern works (grid, zip, custom, mixed)
- **No mode switching**: User controls iteration pattern via Python (list comprehension, itertools, etc.)
- **Natural for samplers**: Creating different samplers per config is trivial
- **Type-safe**: `BSERunConfig` is a proper dataclass with clear fields
- **Testable**: Easy to mock/verify individual configs
- **No eval()**: No string-based parameter assignment

### Cons

- **More verbose for simple cases**: 1D sweep requires a generator function
- **User must understand components**: Need to know about ODE, sampler, solver, etc.

---

## ⭐ Proposal 6: Fixed + Variable Separation (RECOMMENDED)

Builds on Proposal 5 but adds a cleaner abstraction: separate what's **fixed** (base configuration) from what **varies** (parameter variations). The estimator internally generates full configs.

### Core Idea

- Define fixed components once in a `BaseConfig`
- Define variations as lightweight `Variation` objects (only what changes)
- The estimator combines them internally to create full `BSERunConfig` objects
- For complex cases, you can still provide full configs (falls back to Proposal 5)

### API Design

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class BaseConfig:
    """Fixed components that don't change across runs."""
    n: int
    ode_system: ODESystemProtocol
    sampler: Sampler
    solver: SolverProtocol
    feature_extractor: FeatureExtractor
    cluster_classifier: LabelPredictor

    @classmethod
    def from_setup(cls, props: SetupProperties) -> "BaseConfig":
        """Create from setup function output."""
        return cls(
            n=props["n"],
            ode_system=props["ode_system"],
            sampler=props["sampler"],
            solver=props["solver"],
            feature_extractor=props["feature_extractor"],
            cluster_classifier=props["cluster_classifier"],
        )


@dataclass
class Variation:
    """What changes for a single run. Only specify what differs from base."""
    label: dict[str, Any]  # Required: identifies this run, e.g., {"T": 0.5}

    # Optional overrides (None = use base)
    ode_params: dict[str, Any] | None = None  # Updates to ode_system.params
    sampler: Sampler | None = None  # Replace sampler entirely
    solver: SolverProtocol | None = None  # Replace solver entirely
    n: int | None = None  # Override sample count

    # For complex cases: full ODE system replacement
    ode_system: ODESystemProtocol | None = None


class MultiStudyEstimator:
    """Basin stability estimator with fixed base + variable parameters."""

    def __init__(
        self,
        base: BaseConfig,
        variations: list[Variation],
        save_to: str | None = None,
    ):
        self.base = base
        self.variations = variations
        ...

    # Alternative: accept full BSERunConfig list for maximum flexibility
    @classmethod
    def from_configs(
        cls,
        configs: list[BSERunConfig],
        save_to: str | None = None,
    ) -> "MultiStudyEstimator":
        """Create from full config list (Proposal 5 style)."""
        ...

    def estimate(self) -> "MultiStudyResults":
        """Run BSE for each variation."""
        for var in self.variations:
            config = self._build_config(var)
            # Run BSE with config...
        ...

    def _build_config(self, var: Variation) -> BSERunConfig:
        """Combine base + variation into full config."""
        ode = var.ode_system or self.base.ode_system
        if var.ode_params and var.ode_system is None:
            ode = copy.deepcopy(self.base.ode_system)
            ode.params.update(var.ode_params)

        return BSERunConfig(
            ode_system=ode,
            sampler=var.sampler or self.base.sampler,
            solver=var.solver or self.base.solver,
            feature_extractor=self.base.feature_extractor,
            cluster_classifier=self.base.cluster_classifier,
            n=var.n or self.base.n,
            label=var.label,
        )
```

### Usage Examples

#### Simple 1D ODE Parameter Sweep (Cleanest Case)

```python
base = BaseConfig.from_setup(setup_pendulum_system())

variations = [
    Variation(label={"T": t}, ode_params={"T": t})
    for t in np.arange(0.01, 0.97, 0.05)
]

estimator = MultiStudyEstimator(base, variations)
results = estimator.estimate()
```

#### With Ground Truth CSV Samplers

```python
base = BaseConfig.from_setup(setup_pendulum_system())

variations = [
    Variation(
        label={"T": t},
        ode_params={"T": t},
        sampler=CsvSampler(f"ground_truth_T_{t:.2f}.csv", ...),
        n=1000,  # Override n to match CSV
    )
    for t in t_values
]

estimator = MultiStudyEstimator(base, variations)
```

#### 2D Grid with Network Regeneration

```python
from itertools import product

base = BaseConfig.from_setup(setup_rossler_network())

variations: list[Variation] = []
for K, p in product(k_values, p_values):
    # Build new network for this p
    graph = nx.watts_strogatz_graph(n=400, k=8, p=p, seed=42)
    edges_i, edges_j = build_edge_arrays(graph)

    # Need full ODE replacement since topology changes
    ode = RosslerNetworkJaxODE({
        **base.ode_system.params,
        "K": K,
        "edges_i": edges_i,
        "edges_j": edges_j,
    })

    variations.append(Variation(
        label={"K": K, "p": p},
        ode_system=ode,  # Full replacement
    ))

estimator = MultiStudyEstimator(base, variations)
```

#### Hyperparameter Study (Varying N and Tolerance)

```python
base = BaseConfig.from_setup(setup_lorenz_system())

variations = [
    Variation(
        label={"n": n, "rtol": rtol},
        n=n,
        solver=JaxSolver(rtol=rtol, atol=1e-9, ...),
    )
    for n, rtol in product([100, 500, 1000], [1e-3, 1e-6, 1e-9])
]

estimator = MultiStudyEstimator(base, variations)
```

#### Fallback to Full Configs (Proposal 5 Style)

```python
# When you need maximum control, use from_configs
configs = make_complex_configs(...)  # Your custom logic
estimator = MultiStudyEstimator.from_configs(configs)
```

### Helper: Generate Variations from Parameter Grid

```python
def grid_variations(
    ode_params: dict[str, np.ndarray],
) -> list[Variation]:
    """Generate variations for Cartesian product of ODE parameters."""
    from itertools import product

    param_names = list(ode_params.keys())
    param_values = list(ode_params.values())

    variations: list[Variation] = []
    for values in product(*param_values):
        params = dict(zip(param_names, values))
        variations.append(Variation(
            label=params.copy(),
            ode_params=params,
        ))
    return variations

# Usage - extremely concise for simple cases
base = BaseConfig.from_setup(setup_pendulum_system())
variations = grid_variations({"T": np.arange(0.01, 0.97, 0.05)})
estimator = MultiStudyEstimator(base, variations)
```

### Comparison: Proposal 5 vs Proposal 6

| Aspect               | Proposal 5 (Full Configs)      | Proposal 6 (Base + Variations)       |
| -------------------- | ------------------------------ | ------------------------------------ |
| Simple 1D ODE sweep  | Verbose (full config per run)  | Concise (just `ode_params`)          |
| Custom samplers      | Natural                        | Natural (set `sampler` in Variation) |
| Network regeneration | Natural                        | Supported via `ode_system` override  |
| Full flexibility     | ✓ (native)                     | ✓ (via `from_configs`)               |
| Learning curve       | Must understand all components | Can start with just `ode_params`     |

### Pros

- **Simple cases are simple**: 1D ODE sweep is just a list of `Variation(label, ode_params)`
- **Progressive complexity**: Start with `ode_params`, add `sampler`/`solver` overrides as needed
- **Clear separation**: Fixed vs variable is explicit in the API
- **Still flexible**: Full config fallback available
- **Backward compatible**: Can implement `ASBasinStabilityEstimator` on top of this

### Cons

- **Two ways to do things**: `Variation` for simple cases, full `BSERunConfig` for complex
- **Slightly more code** in the library (but simpler user code)

---

### Helper Functions for Common Patterns

```python
# Utility functions to reduce boilerplate

def grid_configs(
    base_props: SetupProperties,
    param_updates: dict[str, np.ndarray],  # {"T": t_values, "sigma": sigma_values}
) -> list[BSERunConfig]:
    """Generate grid of configs varying ODE parameters only."""
    from itertools import product

    param_names = list(param_updates.keys())
    param_values = list(param_updates.values())

    configs: list[BSERunConfig] = []
    for values in product(*param_values):
        label = dict(zip(param_names, values))

        # Clone and update ODE params
        ode = copy.deepcopy(base_props["ode_system"])
        for name, val in label.items():
            ode.params[name] = val

        configs.append(BSERunConfig(
            ode_system=ode,
            sampler=base_props["sampler"],
            solver=base_props["solver"],
            feature_extractor=base_props["feature_extractor"],
            cluster_classifier=base_props["cluster_classifier"],
            n=base_props["n"],
            label=label,
        ))

    return configs

# Usage for simple grid
configs = grid_configs(
    base_props=setup_pendulum_system(),
    param_updates={"T": np.arange(0.01, 0.97, 0.05)},
)
```

---

## Proposal 1: Extended `AdaptiveStudyParams` with Multi-Parameter Support

### API Design

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ParameterSpec:
    """Specification for a single parameter to vary."""
    name: str  # e.g., 'ode_system.params["K"]', 'solver.rtol', 'n'
    values: np.ndarray
    type: Literal["ode", "solver", "sampler", "hyper"] = "ode"

class MultiParameterStudyParams(TypedDict):
    """Parameters for multi-dimensional parameter study."""
    parameters: list[ParameterSpec]
    mode: Literal["grid", "zip", "custom"]  # grid = cartesian product, zip = parallel sweep
    # For custom mode: explicit list of parameter combinations
    combinations: list[dict[str, float]] | None
```

### Usage Examples

#### 2D Grid Study (Cartesian Product)

```python
params = MultiParameterStudyParams(
    parameters=[
        ParameterSpec(name='ode_system.params["K"]', values=k_values, type="ode"),
        ParameterSpec(name='ode_system.params["p"]', values=p_values, type="ode"),
    ],
    mode="grid",  # Creates K × p combinations
)
# Total runs: len(k_values) * len(p_values)
```

#### Parallel Sweep (Zip Mode)

```python
params = MultiParameterStudyParams(
    parameters=[
        ParameterSpec(name='ode_system.params["sigma"]', values=sigma_values),
        ParameterSpec(name='ode_system.params["rho"]', values=rho_values),
    ],
    mode="zip",  # sigma[i] with rho[i]
)
# Total runs: len(sigma_values) == len(rho_values)
```

#### Custom Combinations

```python
params = MultiParameterStudyParams(
    parameters=[
        ParameterSpec(name='ode_system.params["K"]', values=np.array([]), type="ode"),
        ParameterSpec(name="network_seed", values=np.array([]), type="hyper"),
    ],
    mode="custom",
    combinations=[
        {"K": 0.1, "network_seed": 42},
        {"K": 0.2, "network_seed": 43},
        {"K": 0.1, "network_seed": 44},
        # ... non-grid combinations
    ],
)
```

### Pros

- Backward compatible (can keep old `AdaptiveStudyParams` for 1D)
- Explicit parameter types help with validation
- Flexible modes cover common use cases

### Cons

- `eval()` approach for parameter assignment is fragile
- Doesn't handle parameters that require reconstruction (e.g., new sampler instance)

---

## Proposal 2: Callback-Based Parameter Update

Instead of string-based parameter names, use callbacks for maximum flexibility.

### API Design

```python
from typing import Protocol, Any

class ParameterUpdater(Protocol):
    """Protocol for updating parameters before each BSE run."""
    def __call__(
        self,
        ode_system: ODESystemProtocol,
        sampler: Sampler,
        solver: SolverProtocol,
        feature_extractor: FeatureExtractor,
        cluster_classifier: LabelPredictor,
        n: int,
        param_values: dict[str, Any],
    ) -> tuple[ODESystemProtocol, Sampler, SolverProtocol, FeatureExtractor, LabelPredictor, int]:
        """Return updated components for this parameter combination."""
        ...

@dataclass
class ParameterStudyConfig:
    """Configuration for multi-parameter study."""
    parameter_grid: dict[str, np.ndarray]  # name -> values
    mode: Literal["grid", "zip"] = "grid"
    updater: ParameterUpdater | None = None  # Custom update logic
```

### Usage Examples

#### Simple ODE Parameter (Auto-Handled)

```python
config = ParameterStudyConfig(
    parameter_grid={
        "T": np.arange(0.01, 0.97, 0.05),
    },
)
# Automatically updates ode_system.params["T"]
```

#### Complex Update with Callback

```python
def rossler_network_updater(
    ode_system, sampler, solver, feature_extractor, cluster_classifier, n, param_values
):
    """Update Rössler network for new (K, p) combination."""
    p = param_values["p"]
    K = param_values["K"]

    # Generate new network topology for this p
    graph = nx.watts_strogatz_graph(n=400, k=8, p=p, seed=42)
    edges_i, edges_j = build_edge_arrays(graph)

    # Create new ODE system with updated topology and K
    new_params = ode_system.params.copy()
    new_params["K"] = K
    new_params["edges_i"] = edges_i
    new_params["edges_j"] = edges_j
    new_ode = RosslerNetworkJaxODE(new_params)

    return new_ode, sampler, solver, feature_extractor, cluster_classifier, n

config = ParameterStudyConfig(
    parameter_grid={
        "K": k_values,
        "p": p_values,
    },
    mode="grid",
    updater=rossler_network_updater,
)
```

### Pros

- Maximum flexibility for complex cases (network regeneration, sampler changes)
- No fragile `eval()` calls
- Type-safe

### Cons

- More verbose for simple cases
- Users need to understand the component structure

---

## Proposal 3: Builder Pattern with Fluent API

### API Design

```python
class ParameterStudyBuilder:
    """Fluent builder for parameter studies."""

    def vary_ode_param(self, name: str, values: np.ndarray) -> "ParameterStudyBuilder":
        """Add ODE parameter to vary."""
        ...

    def vary_solver_param(self, name: str, values: np.ndarray) -> "ParameterStudyBuilder":
        """Add solver parameter to vary (e.g., rtol, atol)."""
        ...

    def vary_n_samples(self, values: np.ndarray) -> "ParameterStudyBuilder":
        """Vary number of samples."""
        ...

    def vary_sampler(self, samplers: list[Sampler]) -> "ParameterStudyBuilder":
        """Use different sampler instances (e.g., for ground truth CSVs)."""
        ...

    def with_custom_updater(self, fn: Callable) -> "ParameterStudyBuilder":
        """Add custom update callback."""
        ...

    def grid_mode(self) -> "ParameterStudyBuilder":
        """Use Cartesian product of all parameters."""
        ...

    def zip_mode(self) -> "ParameterStudyBuilder":
        """Use parallel iteration (all arrays must have same length)."""
        ...

    def build(self) -> "MultiParameterStudy":
        """Build the parameter study configuration."""
        ...
```

### Usage Examples

```python
study = (
    ParameterStudyBuilder()
    .vary_ode_param("K", k_values)
    .vary_ode_param("p", p_values)
    .grid_mode()
    .build()
)

bse = ASBasinStabilityEstimator(
    n=1000,
    ode_system=ode,
    sampler=sampler,
    solver=solver,
    feature_extractor=fe,
    cluster_classifier=cc,
    study=study,  # Replaces as_params
)
```

#### With Ground Truth Samplers (for Testing)

```python
# For cross-validation with MATLAB
samplers = [
    CsvSampler(f"ground_truth_T_{t:.2f}.csv")
    for t in t_values
]

study = (
    ParameterStudyBuilder()
    .vary_ode_param("T", t_values)
    .vary_sampler(samplers)  # One sampler per T value
    .zip_mode()
    .build()
)
```

### Pros

- Discoverable API (IDE autocomplete)
- Clear separation of parameter types
- Easy to extend

### Cons

- More complex implementation
- May be overkill for simple 1D studies

---

## Proposal 4: Separate Classes for Different Study Types

### API Design

```python
# Keep existing for 1D studies
class ASBasinStabilityEstimator:
    """Single-parameter adaptive study."""
    ...

# New class for multi-parameter
class GridStudyEstimator:
    """Multi-parameter grid study (Cartesian product)."""

    def __init__(
        self,
        base_config: dict,  # n, ode_system, sampler, solver, etc.
        parameter_axes: dict[str, ParameterAxis],
    ):
        ...

@dataclass
class ParameterAxis:
    """One axis of the parameter grid."""
    values: np.ndarray
    update_fn: Callable[[Any, float], Any] | None = None  # How to apply this parameter
    type: Literal["ode", "solver", "sampler", "n"] = "ode"
```

### Usage

```python
estimator = GridStudyEstimator(
    base_config={
        "n": 1000,
        "ode_system": ode,
        "sampler": sampler,
        "solver": solver,
        "feature_extractor": fe,
        "cluster_classifier": cc,
    },
    parameter_axes={
        "K": ParameterAxis(values=k_values, type="ode"),
        "p": ParameterAxis(
            values=p_values,
            update_fn=lambda ode, p: rebuild_network(ode, p),  # Custom logic
        ),
    },
)

results = estimator.run()  # Returns GridStudyResults with 2D indexing
```

### Pros

- Clean separation of concerns
- `GridStudyResults` can have 2D/ND indexing for easy analysis
- No breaking changes to existing API

### Cons

- Users need to learn new class
- Some code duplication between classes

---

## Recommendation

**Proposal 6 (Fixed + Variable Separation)** is the recommended approach because:

1. **Simple cases are simple**: 1D ODE sweep is just `Variation(label, ode_params)` - no full config needed
2. **Clear mental model**: "Here's what's fixed, here's what varies"
3. **Progressive complexity**: Start simple, add overrides (sampler, solver) as needed
4. **Full flexibility available**: `from_configs()` falls back to Proposal 5 for complex cases
5. **Natural sampler support**: Different samplers per variation is straightforward
6. **No eval() hacks**: Type-safe dataclasses throughout

**When to use Proposal 5 directly**: When every run needs completely different components (rare) or when building reusable config generators.

### API Summary

```python
# Simple: 1D ODE parameter sweep
base = BaseConfig.from_setup(setup_pendulum_system())
variations = [Variation(label={"T": t}, ode_params={"T": t}) for t in t_values]
estimator = MultiStudyEstimator(base, variations)

# With custom samplers (for testing)
variations = [
    Variation(label={"T": t}, ode_params={"T": t}, sampler=CsvSampler(f"gt_T_{t}.csv"))
    for t in t_values
]

# Complex: Full config control (Proposal 5 style)
configs = [BSERunConfig(...) for ...]
estimator = MultiStudyEstimator.from_configs(configs)
```

### Migration Path

1. Keep `ASBasinStabilityEstimator` for backward compatibility
2. Implement `BaseConfig`, `Variation`, and `MultiStudyEstimator`
3. Add `BSERunConfig` and `from_configs()` for full flexibility
4. Add helpers: `grid_variations()`, `zip_variations()`
5. Update plotters to work with `MultiStudyResults`

---

## Test Helper Implications

With Proposal 6, the test helper uses variations with custom samplers:

```python
def run_multi_parameter_test(
    json_path: Path,
    base: BaseConfig,
    variation_generator: Callable[[list[float]], list[Variation]],
    z_threshold: float = 2.0,
) -> tuple[MultiStudyEstimator, list[ComparisonResult]]:
    """Run multi-parameter test with ground truth validation."""
    with open(json_path) as f:
        expected_results = json.load(f)

    parameter_values = [r["parameter"] for r in expected_results]
    variations = variation_generator(parameter_values)

    estimator = MultiStudyEstimator(base, variations)
    results = estimator.estimate()

    # Validate against expected_results...
    return estimator, comparison_results
```

Usage with ground truth CSVs:

```python
def make_test_variations(t_values: list[float]) -> list[Variation]:
    return [
        Variation(
            label={"T": t},
            ode_params={"T": t},
            sampler=CsvSampler(f"ground_truth_T_{t:.2f}.csv", ...),
        )
        for t in t_values
    ]

# In test
base = BaseConfig.from_setup(setup_pendulum_system())
estimator, results = run_multi_parameter_test(
    json_path=Path("expected_T_study.json"),
    base=base,
    variation_generator=make_test_variations,
)
```

Alternative with Proposal 5 style (full configs):

```python
def run_multi_parameter_test(
    json_path: Path,
    config_generator: Callable[[list[float]], list[BSERunConfig]],
    z_threshold: float = 2.0,
) -> tuple[MultiStudyEstimator, list[ComparisonResult]]:
    """Run multi-parameter test with ground truth validation.

    The config_generator receives parameter values from JSON and returns
    BSERunConfig list - this allows using CsvSampler per parameter.

    """
    with open(json_path) as f:
        expected_results = json.load(f)

    parameter_values = [r["parameter"] for r in expected_results]

    # User provides generator that can use CsvSampler per param
    configs = config_generator(parameter_values)

    estimator = MultiStudyEstimator(configs)
    results = estimator.estimate()

    # Validate against expected_results...
    return estimator, comparison_results
```

Usage with ground truth CSVs:

```python
def make_test_configs(t_values: list[float]) -> list[BSERunConfig]:
    configs: list[BSERunConfig] = []
    for t in t_values:
        props = setup_pendulum_system(T=t)
        sampler = CsvSampler(f"ground_truth_T_{t:.2f}.csv", ...)
        configs.append(BSERunConfig(
            ode_system=props["ode_system"],
            sampler=sampler,
            ...
            label={"T": t},
        ))
    return configs

# In test
estimator, results = run_multi_parameter_test(
    json_path=Path("expected_T_study.json"),
    config_generator=make_test_configs,
)
```

---

## Next Steps

1. **Implement `BSERunConfig` dataclass** and `MultiStudyEstimator`
2. **Add `MultiStudyResults`** with DataFrame conversion and 2D array reshaping
3. **Create helper functions**: `grid_configs()`, `zip_configs()`
4. **Update plotting** to support `MultiStudyResults`
5. **Migrate `main_rossler_network_2_dimensional.py`** to use new API
6. **Update test helpers** to use config generator pattern

---

## Open Questions

1. Should `ASBasinStabilityEstimator` be deprecated in favor of the new class?
2. Should we support async/parallel execution across configurations?
3. What plotting utilities are needed for 2D results (heatmaps, contour plots)?
