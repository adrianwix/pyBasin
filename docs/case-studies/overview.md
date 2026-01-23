# Case Studies Overview

This section documents the case studies used to validate pyBasin against the original MATLAB bSTAB implementation.

## Classification Quality Metrics

All case study comparisons evaluate classification quality by comparing predicted labels from pyBasin against ground truth labels from MATLAB bSTAB on identical initial conditions.

### Methodology

Since both implementations classify trajectories into attractor labels (e.g., "FP", "LC", "chaos"), we use standard classification metrics to validate that pyBasin correctly replicates MATLAB bSTAB's behavior.

For each test case, we:

1. Load exact initial conditions from MATLAB ground truth CSV files
2. Run pyBasin classification on those same initial conditions
3. Compare predicted labels against MATLAB's ground truth labels
4. Compute classification metrics

### Metrics Used

#### 1. F1-Score (Per Class)

The [F1-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) measures classification quality for each attractor type:

$$F1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

where TP = true positives, FP = false positives, FN = false negatives for that class.

**Range:** [0, 1], where 1.0 = perfect classification for that class

#### 2. Macro F1-Score (Overall)

The macro-averaged F1-score summarizes overall classification quality:

$$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^{K} F1_k$$

where $K$ is the number of classes (attractor types).

**Range:** [0, 1], where 1.0 = perfect classification across all classes

#### 3. Matthews Correlation Coefficient (MCC)

The [Matthews correlation coefficient](https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef) is a global metric measuring correlation between predictions and ground truth.

For **binary classification**:

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

For **multiclass classification**, scikit-learn uses a generalization based on the confusion matrix $C$:

$$\text{MCC} = \frac{c \cdot s - \sum_k p_k \cdot t_k}{\sqrt{(s^2 - \sum_k p_k^2)(s^2 - \sum_k t_k^2)}}$$

where:

- $t_k = \sum_i^K C_{ik}$ — the number of times class $k$ truly occurred
- $p_k = \sum_i^K C_{ki}$ — the number of times class $k$ was predicted
- $c = \sum_k^K C_{kk}$ — the total number of samples correctly predicted
- $s = \sum_i^K \sum_j^K C_{ij}$ — the total number of samples

**Range:** [-1, 1], where:

- +1 = perfect prediction
- 0 = random prediction
- -1 = complete disagreement

MCC is particularly useful for imbalanced datasets (common in basin stability with dominant attractors).

### Quality Thresholds

| Metric       | Excellent | Good   | Acceptable | Poor   |
| ------------ | --------- | ------ | ---------- | ------ |
| **F1**       | ≥ 0.95    | ≥ 0.90 | ≥ 0.80     | < 0.80 |
| **Macro F1** | ≥ 0.95    | ≥ 0.90 | ≥ 0.80     | < 0.80 |
| **MCC**      | ≥ 0.90    | ≥ 0.80 | ≥ 0.70     | < 0.70 |

**Interpretation:**

- **Excellent/Good**: Confirms correct implementation - pyBasin matches MATLAB behavior
- **Acceptable**: Minor discrepancies, may be due to numerical differences or edge cases
- **Poor**: Significant differences requiring investigation

### Reading Comparison Tables

The comparison tables in each case study show:

- **Attractor**: The attractor type (e.g., "FP", "LC", "chaos")
- **pyBasin BS ± SE**: Python implementation basin stability with standard error
- **bSTAB BS ± SE**: MATLAB reference basin stability with standard error
- **F1**: F1-score for this specific attractor class (classification quality)
- **MCC**: Matthews correlation coefficient (same value repeated for all rows - it's a global metric)

Additionally, a summary section displays:

- **Macro F1-score**: Overall classification quality across all attractors
- **Matthews Correlation Coefficient**: Global classification correlation

## Purpose

The case studies serve multiple purposes:

1. **Validation**: Verify correctness by comparing results with MATLAB bSTAB
2. **Examples**: Demonstrate usage patterns for different types of systems
3. **Thesis Artifacts**: Generate figures and results for the bachelor thesis
4. **Benchmarking**: Compare performance and accuracy

## Available Case Studies

### [Duffing Oscillator](duffing.md)

A forced Duffing oscillator exhibiting bistability with two coexisting attractors.

**Key Features:**

- Five coexisting limit cycle attractors
- Supervised vs. unsupervised classification comparison
- Feature extraction using maximum and standard deviation

**Reference:** Thomson, J. M. T., & Stewart, H. B. (2002). _Nonlinear dynamics and chaos_ (2nd ed.). Wiley. (See p. 9, Fig. 1.9)

**Files:** `case_studies/duffing_oscillator/`

---

### [Lorenz System](lorenz.md)

A version of the Lorenz system exhibiting two stable co-existing chaotic attractors.

**Key Features:**

- Two coexisting chaotic attractors and unbounded solutions
- Parameter sweep study (σ)
- Sample size (N) convergence study
- Tolerance (rtol/atol) sensitivity study

**Reference:** Li, C., & Sprott, J. C. (2014). Multistability in the Lorenz system: A broken butterfly. _International Journal of Bifurcation and Chaos_, _24_(10), Article 1450131. https://doi.org/10.1142/S0218127414501314

**Files:** `case_studies/lorenz/`

---

### [Pendulum](pendulum.md)

A forced pendulum system with different forcing parameters.

**Key Features:**

- Multiple parameter cases
- Fixed point and limit cycle attractors
- Supervised classification approach

**Reference:** Menck, P., Heitzig, J., Marwan, N., & Kurths, J. (2013). How basin stability complements the linear-stability paradigm. _Nature Physics_, _9_, 89–92. https://doi.org/10.1038/nphys2516

**Files:** `case_studies/pendulum/`

---

### [Friction System](friction.md)

A mechanical system with friction effects.

**Key Features:**

- Fixed point and limit cycle attractors
- Non-smooth dynamics with friction
- Driving velocity (v_d) parameter sweep study

**Reference:** Stender, M., Hoffmann, N., & Papangelo, A. (2020). The basin stability of bi-stable friction-excited oscillators. _Lubricants_, _8_(12), Article 105. https://doi.org/10.3390/lubricants8120105

**Files:** `case_studies/friction/`

---

### [Rössler Network](rossler-network.md)

A network of coupled Rössler oscillators exhibiting synchronization dynamics.

**Key Features:**

- Coupled oscillator dynamics
- Synchronization analysis
- Network-based basin stability

**Reference:** Menck, P., Heitzig, J., Marwan, N., & Kurths, J. (2013). How basin stability complements the linear-stability paradigm. _Nature Physics_, _9_, 89–92. https://doi.org/10.1038/nphys2516

**Files:** `case_studies/rossler/`

## Running Case Studies

All case studies can be run from the project root:

```bash
# Navigate to project root
cd /path/to/pyBasinWorkspace

# Run a specific case study
uv run python -m case_studies.duffing_oscillator.main_supervised
uv run python -m case_studies.lorenz.main_lorenz
uv run python -m case_studies.pendulum.main_case1
```

## Integration Tests

The case studies are also converted into integration tests that automatically validate correctness:

```bash
# Run all integration tests
uv run pytest tests/integration/

# Run specific case study test
uv run pytest tests/integration/test_duffing.py
```

## Generated Artifacts

Case studies save their outputs to:

- **Figures**: `docs/assets/` — Generated plots and visualizations
- **Results**: `artifacts/results/` — Numerical results (JSON, CSV)

To generate new artifacts, run the integration tests with the `--generate-artifacts` flag:

```bash
# Generate artifacts for all case studies
uv run pytest tests/integration/ --generate-artifacts

# Generate artifacts for a specific case study
uv run pytest tests/integration/test_duffing.py --generate-artifacts
```

## Contributing New Case Studies

To add a new case study:

1. Create a new directory under `case_studies/`
2. Implement the ODE system and feature extractor
3. Create a main script that runs the analysis
4. Add corresponding integration test
5. Document in this section

See [Contributing Guide](../development/contributing.md) for details.
