# Case Studies Overview

This section documents the case studies used to validate pyBasin against the original MATLAB bSTAB implementation.

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
- Two stable periodic attractors
- Parameter-dependent basin stability
- Supervised and unsupervised learning approaches

**Files:** `case_studies/duffing_oscillator/`

---

### [Lorenz System](lorenz.md)

The classic Lorenz system with parameter variations.

**Key Features:**
- Chaotic dynamics
- Parameter sensitivity studies (σ, ρ)
- High-dimensional state space

**Files:** `case_studies/lorenz/`

---

### [Pendulum](pendulum.md)

A forced pendulum system with different forcing parameters.

**Key Features:**
- Multiple parameter cases
- Bifurcation analysis
- Grid-based sampling comparison

**Files:** `case_studies/pendulum/`

---

### [Friction System](friction.md)

A mechanical system with friction effects.

**Key Features:**
- Non-smooth dynamics
- Velocity-dependent friction
- Parameter variation studies

**Files:** `case_studies/friction/`

## Running Case Studies

All case studies can be run from the project root:

```bash
# Navigate to project root
cd /path/to/pyBasinWorkspace

# Run a specific case study
uv run python case_studies/duffing_oscillator/main_supervised.py
uv run python case_studies/lorenz/main_lorenz.py
uv run python case_studies/pendulum/main_case1.py
```

## Comparison with MATLAB

Each case study includes validation against the MATLAB implementation:

| Case Study | MATLAB BS | Python BS | Difference |
|------------|-----------|-----------|------------|
| Duffing (Case 1) | TBD | TBD | TBD |
| Lorenz (σ=10) | TBD | TBD | TBD |
| Pendulum (Case 1) | TBD | TBD | TBD |
| Friction (v=1) | TBD | TBD | TBD |

*Values to be filled after validation runs*

## Generated Artifacts

All case studies save their outputs to the `artifacts/` directory:

```
artifacts/
├── figures/          # Generated plots
├── results/          # Numerical results (JSON, CSV)
└── reports/          # Summary reports
```

## Integration Tests

The case studies are also converted into integration tests that automatically validate correctness:

```bash
# Run all integration tests
pytest tests/integration/

# Run specific case study test
pytest tests/integration/test_duffing.py
```

## Contributing New Case Studies

To add a new case study:

1. Create a new directory under `case_studies/`
2. Implement the ODE system and feature extractor
3. Create a main script that runs the analysis
4. Add corresponding integration test
5. Document in this section

See [Contributing Guide](../development/contributing.md) for details.
