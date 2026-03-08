# Parameter Batching Architecture

## Problem Statement

`BasinStabilityStudy.run()` iterates serially over P parameter combinations\_. At each
step it creates a fresh `BasinStabilityEstimator`, integrates N initial conditions,
extracts features, classifies, and computes basin stability. The integration step
dominates wall-clock time, and the serial loop leaves hardware (GPU / multi-core CPU)
underutilized.

The experiment in `experiments/solver/experiment_parameter_batching.py` confirms that
both Diffrax (JAX vmap) and torchdiffeq (flattened P\*B batch) can integrate multiple
parameter combinations in a single call. With the `t_eval` save-window logic already
in place, memory consumption stays manageable -- the machine handles 100k trajectories
in one batch when only the steady-state window is stored.

The goal: batch the **integration** of P parameter sets x B initial conditions into
one solver call (or a few sub-batches), then split trajectories back per parameter
for feature extraction, classification, and BS computation.

A secondary goal is to **unify the return type**: `BasinStabilityEstimator.estimate_bs()`
currently returns `dict[str, float]` while the Study wraps that into `StudyResult`.
Both paths should produce `StudyResult` directly.

## Current Pipeline (per parameter combination)

```
Sampling  -->  Integration  -->  Solution  -->  Features  -->  Classification  -->  BS
   B ICs       (B, N, S)        object         (B, F)           (B,) labels       dict
```

Feature extraction and classification are inherently **per-parameter**: different
parameter values yield different attractor landscapes, so features and labels are
only meaningful within a single parameter set. Only the integration step can truly
benefit from cross-parameter batching.

## Architectural Options

### Option A -- Batched solver method (recommended)

Add a second entry point to the solver that accepts a parameter grid alongside the
initial conditions and returns trajectories indexed by parameter set.

**Solver-level change:**

```python
# New method on SolverProtocol / Solver base / JaxSolver
def integrate_batched(
    self,
    ode_system: ODESystemProtocol,
    y0: Tensor,                    # (B, S)  shared ICs
    param_grid: Tensor | Array,    # (P, n_params)
) -> list[tuple[Tensor, Tensor]]:  # P x (t, y) where y is (N, B, S)
```

Inside the implementation:

| Backend     | Strategy                                                          |
| ----------- | ----------------------------------------------------------------- |
| JaxSolver   | Flatten P\*B; pass params as traced arrays in a single vmap call. |
| TorchDiffEq | Flatten P\*B; wrap ODE with per-sample parameter vectors.         |

After integration, the flat `(N, P*B, S)` output is reshaped to P groups of `(N, B, S)`.

**ODE system change -- new protocol method:**

The ODE system needs a way to evaluate with externally supplied parameter values
instead of reading from `self.params`. One clean approach:

```python
class ODESystemProtocol(Protocol):
    ...
    def ode_with_params(self, t, y, params_vector) -> Tensor:
        """Evaluate the ODE with an explicit parameter vector (for batched solving)."""
```

Alternatively, avoid protocol changes entirely: the solver can temporarily overwrite
`ode_system.params` per group, but that only works for the serial fallback. For true
P\*B flattening, the ODE callable must receive params as an argument.

**Study-level change:**

`BasinStabilityStudy.run()` collects all parameter combinations from `StudyParams`,
builds the `(P, n_params)` grid, calls `solver.integrate_batched(...)` once, then
loops over the P trajectory groups for features + classification.

**Pros:**

- Batching logic is encapsulated in the solver -- the rest of the pipeline stays
  per-parameter.
- Both JAX and torchdiffeq support this pattern (confirmed experimentally).
- Sub-batching for memory limits (`max_batch_size`) fits naturally inside the solver.
- Template integration for supervised classifiers is unaffected (small batch, separate
  call).

**Cons:**

- Requires a new `ode_with_params` protocol method on every ODE system.
- Caching becomes trickier (cache key must include the parameter vector).
- torchdiffeq wrapper needs a `BatchedODE` adapter class per ODE, or a generic one
  that maps parameter indices.

### Option B -- Study-level orchestration (no solver API change)

Keep the solver interface unchanged. Instead, `BasinStabilityStudy` builds
P copies of the ODE system (one per parameter combination), concatenates their y0
into a `(P*B, S)` tensor, calls `solver.integrate()` once with a special
"multi-param ODE" wrapper, then slices the result.

**How it works:**

```python
# Study builds wrapper that dispatches per-sample params
class _MultiParamODE:
    def __init__(self, ode_system, param_sets, B):
        ...
    def ode(self, t, y):  # y is (P*B, S)
        # each row i uses params[i // B]
```

**Pros:**

- Zero changes to `Solver`, `SolverProtocol`, or `ODESystemProtocol`.
- All batching logic isolated in `BasinStabilityStudy`.

**Cons:**

- The Study must understand ODE internals (how to map parameter dicts to vectors).
- The multi-param ODE wrapper is fragile: different ODE systems have different
  parameter structures.
- Only works for the Study path; standalone BSE gains nothing.
- Harder to sub-batch if P\*B exceeds memory.

### Option C -- New `BatchedBasinStabilityEstimator` class

Create a new class that accepts P parameter sets + shared configuration and returns
`list[StudyResult]`. Internally it does:

1. Sample B ICs (once, shared).
2. Integrate P\*B trajectories in one call (like Option A).
3. Split trajectories into P groups of `(N, B, S)`.
4. For each group: build Solution, extract features, classify, compute BS.
5. Return `list[StudyResult]`.

```python
class BatchedBasinStabilityEstimator:
    def __init__(self, ..., param_grid: Tensor): ...
    def estimate_bs(self) -> list[StudyResult]: ...
```

`BasinStabilityStudy` becomes a thin wrapper that constructs a
`BatchedBasinStabilityEstimator` and collects results.

**Pros:**

- Clean separation: existing BSE is untouched.
- Reusable outside the Study (e.g., custom scripts).
- Return type naturally aligns with `StudyResult`.

**Cons:**

- Duplicates substantial BSE logic (feature extraction, classification, BS
  computation, error calculation, orbit data, unbounded detection).
- Two classes to maintain and keep in sync.
- Still needs the `ode_with_params` mechanism from Option A at the solver level.

### Option D -- Extend BSE with optional parameter grid

Add a `param_grid` argument to `BasinStabilityEstimator.estimate_bs()`:

```python
def estimate_bs(self, param_grid=None) -> StudyResult | list[StudyResult]:
```

When `param_grid` is None, single-parameter mode (current behavior, returns one
`StudyResult`). When provided, batched mode (returns list).

**Pros:**

- No new classes; minimal API surface.
- Single place to maintain the pipeline.

**Cons:**

- Makes BSE significantly more complex (branching everywhere).
- Return type varies by call -- awkward for callers.
- Hard to test and reason about the two code paths.

## Comparison Summary

| Criterion                | A (solver method) | B (study-level) | C (new class)  | D (extend BSE) |
| ------------------------ | ----------------- | --------------- | -------------- | -------------- |
| Solver API change        | yes               | no              | yes            | yes            |
| ODE protocol change      | yes               | no              | yes            | yes            |
| BSE changes              | minimal           | none            | none           | heavy          |
| Study changes            | moderate          | heavy           | moderate       | moderate       |
| Code duplication         | low               | medium          | high           | low            |
| Works for standalone BSE | no (study only)   | no              | yes            | yes            |
| Memory sub-batching      | solver handles    | tricky          | solver handles | solver handles |
| Return type unification  | orthogonal        | orthogonal      | built-in       | built-in       |

## Design Principle: P=1 is not a special case

The distinction between "single parameter" and "batched parameters" should not
exist in the API. Every integration call carries a parameter array. When
`BasinStabilityEstimator` runs standalone, that array has length 1. When
`BasinStabilityStudy` runs, `StudyParams` produces the full array (via grid,
zip, or custom list). The solver sees the same interface either way.

This collapses Options A--D into a single unified design with no branching,
no `integrate_batched` vs `integrate`, and no conditional return types.

## Recommendation

### Unified pipeline

```
params: (P, n_ode_params)       -- P=1 for standalone BSE, P>1 for study
y0:     (B, S)                  -- shared ICs, sampled once

Integration  -->  (P, N, B, S)  or equivalently P x (N, B, S) after split
     |
     v  (per parameter set p)
Solution_p  -->  Features_p  -->  Classification_p  -->  StudyResult_p
```

The solver flattens P\*B internally, integrates once, reshapes to P groups.
Everything downstream (Solution, features, classification, BS) runs per group.
Both BSE and Study return `list[StudyResult]` -- length 1 for standalone,
length P for a study.

### Track 1 -- Unify return type (small, do first)

1. Change `BasinStabilityEstimator.estimate_bs()` to return `StudyResult`.
2. Move error computation, label collection, and orbit data assembly into
   `estimate_bs` so the result is self-contained.
3. Simplify `BasinStabilityStudy.run()` -- it no longer needs to manually build
   `StudyResult` dicts; it just collects what BSE returns.
4. Update `save()` methods and tests accordingly.

### Track 2 -- ODE system redesign and parameter-aware integration

The current ODE systems read parameters from `self.params` (a dict on the
instance). This makes parameter variation awkward: you cannot vmap over a dict
read in JAX, and torchdiffeq needs adapter classes to inject per-sample
parameters. Adding a separate `ode_with_params` method alongside the existing
`ode` would be a patch, not a fix.

The clean solution: like Julia's SciML, **parameters are always a function
argument**. The ODE signature becomes `ode(t, y, p)` where `p` is a flat
array. The instance holds default parameters (as a typed dict for
construction and human readability) and structural state (topology, constants
that never change). The solver always passes `p` explicitly.

#### New ODE base classes

**JAX:**

```python
class JaxODESystem[P]:
    PARAM_KEYS: ClassVar[tuple[str, ...]]   # ordered dict -> array mapping

    def __init__(self, params: P):
        self.default_params: P = params

    @abstractmethod
    def ode(self, t: Array, y: Array, p: Array) -> Array:
        """RHS. p is shape (..., n_params) -- always passed by the solver."""
        ...

    def params_to_array(self, params: P | None = None) -> Array:
        p = params or self.default_params
        return jnp.array([p[k] for k in self.PARAM_KEYS])
```

**PyTorch:**

```python
class ODESystem[P](ABC, nn.Module):
    PARAM_KEYS: ClassVar[tuple[str, ...]]

    def __init__(self, params: P):
        super().__init__()
        self.default_params: P = params

    @abstractmethod
    def ode(self, t: Tensor, y: Tensor, p: Tensor) -> Tensor:
        ...

    def params_to_array(self, params: P | None = None) -> Tensor:
        p = params or self.default_params
        return torch.tensor([p[k] for k in self.PARAM_KEYS])
```

Key properties:

- `PARAM_KEYS` declares which dict keys map to the flat array and in what
  order. This is the contract between the dict world (construction, study
  params, human-readable labels) and the array world (solver, vmap, batching).
- `default_params` replaces `self.params`. The name makes it clear this is
  the fallback, not the only source of parameter values.
- `params_to_array()` converts the dict to a flat array. The solver calls
  this when `params=None` (standalone BSE, P=1).
- There is no separate `ode_with_params` method. There is only `ode(t, y, p)`.

#### Example: Duffing oscillator (JAX)

```python
class DuffingJaxODE(JaxODESystem[DuffingParams]):
    PARAM_KEYS = ("delta", "k3", "A")

    def ode(self, t: Array, y: Array, p: Array) -> Array:
        delta, k3, A = p[..., 0], p[..., 1], p[..., 2]

        x = y[..., 0]
        x_dot = y[..., 1]

        dx_dt = x_dot
        dx_dot_dt = -delta * x_dot - k3 * x**3 + A * jnp.cos(t)
        return jnp.stack([dx_dt, dx_dot_dt], axis=-1)
```

The body is nearly identical to today. The only change: `self.params["delta"]`
becomes `p[..., 0]`. The `[..., i]` indexing works at any batch dimension --
scalar `(n_params,)`, per-IC `(B, n_params)`, or flattened `(P*B, n_params)`.

#### Example: Rössler network (structural + study params)

Structural state (topology, node count) stays on `self`. Only the float
parameters that a study might sweep go into `p`:

```python
class RosslerNetworkJaxODE(JaxODESystem[RosslerNetworkParams]):
    PARAM_KEYS = ("a", "b", "c", "K")   # only the float params

    def __init__(self, params: RosslerNetworkParams):
        super().__init__(params)
        # Structural -- never varies in a study
        self._N = params["N"]
        self._edges_i = params["edges_i"]
        self._edges_j = params["edges_j"]

    def ode(self, t: Array, y: Array, p: Array) -> Array:
        a, b, c, k = p[..., 0], p[..., 1], p[..., 2], p[..., 3]

        n = self._N
        x = y[:n]
        y_state = y[n:2*n]
        z = y[2*n:]

        diff = x[self._edges_i] - x[self._edges_j]
        coupling = jnp.zeros_like(x).at[self._edges_i].add(diff)

        dx_dt = -y_state - z - k * coupling
        dy_dt = x + a * y_state
        dz_dt = b + z * (x - c)
        return jnp.concatenate([dx_dt, dy_dt, dz_dt])
```

`PARAM_KEYS` explicitly excludes `edges_i`, `edges_j`, and `N` -- they are
structural, not study parameters.

#### Solver changes

The `integrate` signature gains an optional `params` argument:

```python
def integrate(
    self,
    ode_system: ODESystemProtocol,
    y0: Tensor,                             # (B, S)
    params: Tensor | Array | None = None,   # (P, n_params) or None for P=1
) -> tuple[Tensor, Tensor]:
```

**JaxSolver** -- diffrax already expects `ode(t, y, args)`. The `args`
slot is exactly `p`:

```python
def integrate(self, ode_system, y0, params=None):
    if params is None:
        params = ode_system.params_to_array()   # (n_params,)

    term = ODETerm(ode_system.ode)   # ode(t, y, args) -- args IS p

    if params.ndim == 1:             # P=1
        sol = diffeqsolve(term, ..., y0=y0, args=params)
    else:                            # P>1: vmap over param axis
        solve_one = lambda p: diffeqsolve(term, ..., y0=y0, args=p)
        sol = jax.vmap(solve_one)(params)
```

No adapter, no wrapper. The ODE already has the right signature.

**TorchDiffEqSolver** -- torchdiffeq wants `f(t, y)`, so the solver wraps
to capture `p`:

```python
def integrate(self, ode_system, y0, params=None):
    if params is None:
        params = ode_system.params_to_array()   # (n_params,)

    if params.ndim == 1:             # P=1: broadcast p to all ICs
        def ode_fn(t, y):
            return ode_system.ode(t, y, params)
        return torchdiffeq.odeint(ode_fn, y0, t_span)
    else:                            # P>1: flatten P*B
        P, B = params.shape[0], y0.shape[0]
        y0_flat = y0.repeat(P, 1)                        # (P*B, S)
        p_flat = params.repeat_interleave(B, dim=0)       # (P*B, n_params)

        def ode_fn(t, y):
            return ode_system.ode(t, y, p_flat)
        return torchdiffeq.odeint(ode_fn, y0_flat, t_span)
```

The `[..., i]` indexing in the ODE handles both cases automatically:
`(n_params,)` broadcasts with `(B, S)`, and `(P*B, n_params)` matches
`(P*B, S)`.

#### StudyParams produces the parameter array

The study collects all `RunConfig` objects, extracts the ODE parameter
values, and stacks them into a `(P, n_params)` array using `PARAM_KEYS`
for column ordering:

```python
param_grid = jnp.stack([
    jnp.array([run_config.param_values[k] for k in ode_system.PARAM_KEYS])
    for run_config in study_params
])  # (P, n_params)
```

The clean dict contract (`{"T": [...], "K": [...]}`) maps naturally to
column indices via `PARAM_KEYS`.

#### BasinStabilityStudy.run() becomes

1. Sample B ICs once.
2. Build `(P, n_params)` from `StudyParams` + `PARAM_KEYS`.
3. Call `solver.integrate(ode_system, y0, params=param_grid)`.
4. Reshape `(N, P*B, S)` into P groups of `(N, B, S)`.
5. For each group: build Solution, extract features, classify, compute BS.
6. Collect `list[StudyResult]`.

Templates are integrated once (fixed params) and reused across all P groups.

#### Caching

When `params` is provided, the cache key includes the full parameter array.
At P=1 with `params=None` the key is the same as today (derived from
`ode_system.get_str()` which uses `default_params`).

#### Summary of changes vs current design

| Aspect                       | Current                             | New                                   |
| ---------------------------- | ----------------------------------- | ------------------------------------- |
| ODE signature                | `ode(t, y)` (+ `args` for JAX)      | `ode(t, y, p)` always                 |
| Param source in ODE body     | `self.params["delta"]` (dict read)  | `p[..., 0]` (array index)             |
| Param storage                | `self.params` (mutable dict)        | `self.default_params` (fallback)      |
| Dict-to-array mapping        | Not defined                         | `PARAM_KEYS` + `params_to_array()`    |
| JAX vmap over params         | Impossible (can't trace dict reads) | Natural (`args=p`, vmap over p)       |
| PyTorch batched params       | Needs `_BatchedODE` adapter         | Closure captures `p_flat`             |
| Separate method for batching | `ode_with_params` patch             | Gone -- there is only `ode(t, y, p)`  |
| Structural state             | Mixed into `self.params`            | On `self`, excluded from `PARAM_KEYS` |

## Plotter Compatibility Constraint

Both `MatplotlibStudyPlotter` and the interactive `InteractivePlotter` depend on
`BasinStabilityStudy`. Whatever the final batched design looks like, the plotters
must keep working without a rewrite. The contract they rely on is documented below.

### What MatplotlibStudyPlotter consumes

Pure data reader -- it never re-runs estimation. It accesses:

- `bs_study.results: list[StudyResult]` -- iterates, indexes, reads length.
- `bs_study.studied_parameter_names: list[str]` -- for axis labels and grouping.
- `bs_study.output_dir` -- for saving figures.
- Per result: `r["study_label"]`, `r["basin_stability"]`, `r["orbit_data"]`,
  `r["labels"]`. Type checks on study_label values to filter non-numeric params.

No other attribute matters for this plotter.

### What InteractivePlotter consumes

More demanding because it **re-creates BSE instances on demand** for interactive
drill-down into individual parameter values.

**Mode detection:**

- `isinstance(bse, BasinStabilityStudy)` -- hard type check to branch into
  parameter-study mode vs single-BSE mode.

**Study-level reads (same as matplotlib plotter):**

- `bs_study.results`, `bs_study.studied_parameter_names`

**For BSE re-creation** (`_compute_param_bse` and `StudyParameterManagerAIO`):

- `bs_study.study_params` -- iterated to get `RunConfig` objects.
- `bs_study.n`, `bs_study.ode_system`, `bs_study.sampler`, `bs_study.solver`,
  `bs_study.feature_extractor`, `bs_study.estimator`,
  `bs_study.template_integrator` -- all placed into a context dict.
- `run_config.assignments` -- each assignment is `exec()`-ed to mutate the
  context (e.g., `ode_system.params["T"] = 0.5`).
- A new `BasinStabilityEstimator` is constructed from the mutated context and
  `estimate_bs()` is called to get full Solution + features + labels.

**Other AIO components:**

- `ParamOverviewAIO` and `ParamOrbitDiagramAIO` read `bs_study.results` and
  `bs_study.studied_parameter_names` (same read-only pattern).
- `ParamOrbitDiagramAIO` caches `orbit_data.peak_values.cpu().numpy()` per
  result index to avoid repeated GPU transfers.

### Implications for the batched design

1. **`results: list[StudyResult]` is the stable interface.** All plotters
   iterate this list. As long as the batched path populates the same list with
   the same `StudyResult` shape, plotting works unchanged.

2. **`studied_parameter_names` must remain available.** Both plotters use it
   for axis labels and grouping logic.

3. **The interactive plotter's BSE re-creation is the hard constraint.** It
   reads `study_params`, `ode_system`, `solver`, and other components directly
   off the study object, then rebuilds a fresh BSE. Two ways to handle this:
   - **(a) Keep `BasinStabilityStudy` as the public-facing class.** Batching
     happens internally (the `.run()` method calls `integrate_batched` under
     the hood). The study object still carries all components. Plotters see no
     change.
   - **(b) Introduce a `StudyResultSet` data object.** Plotters accept this
     instead of a `BasinStabilityStudy`. The `StudyResultSet` holds `results`,
     `studied_parameter_names`, and the original components needed for
     re-creation. Requires updating `isinstance` checks in `InteractivePlotter`
     and type annotations in `MatplotlibStudyPlotter`. More disruptive.

   Option (a) is strongly preferred -- it preserves backward compatibility
   with zero plotter changes.

4. **`StudyResult` shape is already correct.** The `StudyResult` TypedDict
   (`study_label`, `basin_stability`, `errors`, `n_samples`, `labels`,
   `orbit_data`) is what runs need. Both single-BSE-returned results and
   study-collected results should use this same shape.

## Resolved Questions

1. **Templates are fixed across parameter sets.** The `TemplateIntegrator`
   returns training data based on its own initial conditions and ODE params,
   which do not change during the study. Templates can be integrated once and
   reused for all P parameter combinations.

2. **Assume all parameters fit in memory for now.** A separate experiment
   should determine the practical P\*B limit and sub-batching strategy, but
   the initial implementation can skip memory-budget logic.

3. **Integration is always parameter-aware; P=1 is the default.** When
   `params=None` (standalone BSE), the solver uses `ode_system.params` as
   today -- effectively P=1. When `params` is a `(P, n_ode_params)` array
   (study mode), the solver flattens P\*B and integrates in one call. No
   fallback logic needed; the two cases are the same code path.

4. **Feature extraction parallelism is already handled.** `TorchFeatureExtractor`
   manages its own parallelism (CPU multiprocessing or GPU batched CUDA ops).
   No additional `ThreadPoolExecutor` wrapper needed around the per-parameter
   feature extraction loop.

5. **Prefer a clean parameter mapping contract.** A dict like
   `{"T": [0.1, 0.2, ...], "K": [1.0, 1.0, ...]}` is preferable to the
   current string-expression approach. However, the final interface depends
   on implementation details: if the batched solver does not need to know about
   solver hyperparams (rtol, atol, etc.) -- only ODE params -- then the clean
   dict approach works. The current `StudyParams` string expressions
   (`ode_system.params["T"]`) can be parsed to extract parameter names and
   values for the dict.
