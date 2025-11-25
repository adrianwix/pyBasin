# Feature Extraction Map for Trajectory Classification

Assume for each trajectory you have a tail segment:
- State time series: `X(t_k) ∈ R^n`, k = 1,…,M
- Optionally, a scalar observable: `s(t_k)` (e.g. one coordinate)

Below is a structured list of feature families you can implement.  
You do not need all of them at once; they are building blocks.

> **Note:** Features marked with ✅ are available in **tsfresh** (operates on scalar time series).  
> Features marked with ❌ are **NOT in tsfresh** and require custom implementation.

---

## tsfresh Coverage Summary

| Feature Category                 | tsfresh Coverage                  |
| -------------------------------- | --------------------------------- |
| Basic state-space statistics     | ❌ Not covered (multi-dimensional) |
| Time-domain features             | ✅ Mostly covered                  |
| Frequency-domain features        | ✅ Mostly covered                  |
| Dynamical-instability (Lyapunov) | ❌ Not covered                     |
| Dimension estimates              | ❌ Not covered                     |
| Recurrence-based (RQA)           | ❌ Not covered                     |
| Topological (TDA)                | ❌ Not covered                     |

### Recommended Libraries for Non-tsfresh Features
- **nolds** or **neurokit2** — Lyapunov exponents, correlation dimension
- **PyRQA** or **pyrqa** — Recurrence quantification analysis
- **giotto-tda** or **ripser** — Persistent homology (TDA)

---

## 1. Basic state-space statistics ❌

> **Not in tsfresh** — tsfresh operates on scalar time series, not multi-dimensional state vectors.

Input: tail points `X(t_k)` in state space.

Features:
- Mean state vector:  
  - `μ = (1/M) Σ_k X(t_k)`  
  - Separates attractors with different "centers" (different equilibria, spatially separated cycles).
- Per-coordinate standard deviation:  
  - `σ_i = std(X_i(t_k))`  
  - Distinguishes point-like vs oscillatory/chaotic attractors.
- Size/radius measures:  
  - Mean radius: `(1/M) Σ_k ‖X(t_k) − μ‖`  
  - Max radius: `max_k ‖X(t_k) − μ‖`  
  - Captures attractor "extent" in phase space.
- Covariance / PCA eigenvalues:  
  - Eigenvalues of covariance matrix of `X(t_k) − μ`  
  - Describe anisotropy and effective dimensionality of the attractor's shape.

---

## 2. Time-domain features (on one or few coordinates) ✅

> **Mostly in tsfresh** — `mean`, `variance`, `minimum`, `maximum`, `skewness`, `kurtosis`, `autocorrelation`, `partial_autocorrelation`, `agg_autocorrelation`, `number_crossing_m`

Input: scalar time series `s(t_k)` (e.g. one component of `X`).

Features:
- Basic statistics: mean, variance, min, max, range, skewness, kurtosis.
- Autocorrelation values:  
  - Autocorrelation at small lags (e.g. lag 1, lag corresponding to dominant period).  
  - Distinguishes strongly periodic from noisy/chaotic signals.
- Zero-crossing rate, sign-change counts, etc.  
  - Simple measures of oscillatory activity.

---

## 3. Frequency-domain features (FFT-based) ✅

> **Mostly in tsfresh** — `fft_coefficient`, `fft_aggregated`, `fourier_entropy`, `spkt_welch_density`

Input: scalar time series `s(t_k)`.

Features from power spectrum `P(ω)`:
- Dominant frequency index or value:  
  - Location of maximal spectral peak (excluding DC).  
  - Characteristic oscillation frequency.
- Peak power ratio:  
  - `P_max / Σ P`  
  - High for almost pure limit cycles; lower for quasi-periodic/chaotic.
- Spectral entropy:  
  - `−Σ p_i log p_i` with `p_i = P_i / Σ P_i`  
  - Low for concentrated spectra (clean periodic), high for broad spectra (chaotic/noisy).
- Bandpower ratios (optional):  
  - Power in "fundamental band" vs total power.

---

## 4. Dynamical-instability features ❌

> **Not in tsfresh** — Requires specialized algorithms for Lyapunov exponent estimation.  
> **Libraries:** `nolds`, `neurokit2`

Input: trajectories from nearby initial conditions.

Features:
- Largest Lyapunov exponent (LLE):  
  - Positive → sensitive dependence (chaotic attractor).  
  - Zero → neutral (limit cycle, torus).  
  - Negative → strongly attracting fixed point (in the observed direction).
- Vector of Lyapunov exponents (if feasible):  
  - Can be collapsed to scalar summaries (sum of positive exponents, etc.).

---

## 5. Dimension estimates (fractal / effective dimension) ❌

> **Not in tsfresh** — Requires fractal dimension algorithms.  
> **Libraries:** `nolds`, `neurokit2`

Input: tail point cloud `X(t_k)`.

Features:
- Correlation dimension (D₂):  
  - Based on scaling of correlation sum `C(ε)` with ε.  
  - ~0 for point attractor, ~1 for simple cycle, ~2 for torus, non-integer/higher for strange attractors.
- Box-counting dimension:  
  - Scaling of number of occupied boxes vs box size.  
  - Similar interpretation to correlation dimension; often more heuristic numerically.
- Kaplan–Yorke (Lyapunov) dimension:  
  - Constructed from ordered Lyapunov exponents.  
  - Provides an effective attractor dimension from dynamical instability data.

These give scalar or low-dimensional descriptors of "complexity" of the attractor.

---

## 6. Recurrence-based features ❌

> **Not in tsfresh** — Requires recurrence plot construction and RQA algorithms.  
> **Libraries:** `PyRQA`, `pyrqa`

Input: tail point cloud `X(t_k)` or scalar time series `s(t_k)`.

### 6.1 Recurrence plots and RQA (Recurrence Quantification Analysis)

1. Build recurrence matrix:
   - `R_ij = 1` if `‖X(t_i) − X(t_j)‖ < ε`, else 0  
     (or using an embedded scalar time series).
2. From `R`, compute RQA measures:

Typical RQA features:
- Recurrence rate (RR):  
  - Fraction of ones in `R`.  
  - Overall density of revisits to similar states.
- Determinism (DET):  
  - Fraction of recurrence points forming diagonal line structures.  
  - High for regular/periodic dynamics, lower for chaotic/noisy.
- Average diagonal line length (L), maximum diagonal length (L_max):  
  - Related to predictability and divergence rates.
- Laminarity (LAM), trapping time (TT):  
  - Based on vertical line structures in `R`; relate to laminar (intermittent) phases.
- Entropy of diagonal line lengths:  
  - Variability/complexity of recurrence patterns.

These produce a compact vector of scalars describing temporal regularity vs chaos.

### 6.2 Recurrence network features (optional)

Interpret recurrence matrix `R` as adjacency matrix of an undirected graph.

Graph-theoretic features:
- Average degree, degree distribution statistics.
- Clustering coefficient, transitivity.
- Average path length.
- Network assortativity and modularity measures.

These capture geometric and dynamical structure of the attractor through network topology.

---

## 7. Topological (TDA) features ❌

> **Not in tsfresh** — Requires persistent homology computation.  
> **Libraries:** `giotto-tda`, `ripser`, `gudhi`

Input: tail point cloud `X(t_k)`.

Use persistent homology on a point cloud built from `X(t_k)` (e.g. Vietoris–Rips or Čech complexes).

From persistence diagrams (for dimensions 0, 1, possibly 2):

Features:
- Betti numbers at selected scales:  
  - β₀ (number of components), β₁ (number of loops), etc.
- Number of "long-lived" features in each homology dimension.  
  - E.g. 1 strong 1D loop → limit-cycle-like; multiple or complex loops → tori/strange attractors.
- Total persistence in each dimension:  
  - Σ lifetimes of homological features → global measure of topological complexity.
- Maximum persistence (per dimension):  
  - Strength of the most prominent hole/component.

These topological features distinguish attractors by their qualitative shape in phase space.

---

## 8. Suggested modular structure for your toolbox

For each trajectory, you can implement feature extractors grouped as:

| Extractor                      | tsfresh? | Description                                                              |
| ------------------------------ | -------- | ------------------------------------------------------------------------ |
| `TsfreshFeatureExtractor`      | ✅        | Wrapper around tsfresh for time-domain and frequency features            |
| `StateSpaceStatsExtractor`     | ❌        | Mean, std per coordinate; radius; PCA eigenvalues                        |
| `LyapunovFeaturesExtractor`    | ❌        | Largest Lyapunov exponent, optional full spectrum summaries              |
| `DimensionFeaturesExtractor`   | ❌        | Correlation dimension, box-counting dimension, Kaplan–Yorke dimension    |
| `RecurrenceFeaturesExtractor`  | ❌        | RQA measures; optional recurrence-network graph features                 |
| `TopologicalFeaturesExtractor` | ❌        | Persistent-homology-based summaries: Betti counts, total/max persistence |

### Implementation Priority for Attractor Classification

1. **`LyapunovFeaturesExtractor`** — Most critical for distinguishing chaos from periodicity
2. **`DimensionFeaturesExtractor`** — Correlation dimension directly indicates attractor type
3. **`RecurrenceFeaturesExtractor`** — Well-established RQA metrics for dynamical systems
4. **`StateSpaceStatsExtractor`** — Simple but effective for multi-dimensional phase space analysis
5. **`TopologicalFeaturesExtractor`** — Powerful but computationally expensive

You can then concatenate selected feature sets into a final feature vector and feed it to KNN, DBSCAN, or any other classifier/clustering algorithm.
