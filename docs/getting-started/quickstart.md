# Quick Start

This guide will help you get started with pyBasin in just a few minutes.

## Basic Example

Here's a simple example of estimating basin stability for a 2D dynamical system:

```python
import numpy as np
from pybasin import BasinStabilityEstimator, ODESystem

# Step 1: Define your dynamical system
class SimpleSystem(ODESystem):
    """A simple 2D system with two stable fixed points."""

    def dynamics(self, t, state):
        """Define the differential equations."""
        x, y = state
        dx = x * (1 - x**2 - y**2)
        dy = y * (1 - x**2 - y**2)
        return np.array([dx, dy])

    def classify_attractor(self, solution):
        """Classify which attractor the solution reached."""
        final_state = solution.y[:, -1]
        x_final, y_final = final_state

        # Classify based on final position
        if x_final > 0:
            return 0  # Right attractor
        else:
            return 1  # Left attractor

# Step 2: Create the estimator
system = SimpleSystem()
estimator = BasinStabilityEstimator(
    system=system,
    t_span=(0, 50),  # Integration time
    n_samples=1000   # Number of initial conditions
)

# Step 3: Define the sampling region
bounds = [(-2, 2), (-2, 2)]  # [x_min, x_max], [y_min, y_max]

# Step 4: Estimate basin stability
results = estimator.estimate(bounds)

# Step 5: Analyze results
print(f"Basin Stability (Attractor 0): {results.basin_stability[0]:.3f}")
print(f"Basin Stability (Attractor 1): {results.basin_stability[1]:.3f}")
print(f"Total samples: {results.n_samples}")
print(f"Attractor distribution: {results.attractor_counts}")

# Step 6: Visualize
results.plot_basin_2d()
```

## Using Adaptive Sampling

For more efficient sampling, use the adaptive sampling estimator:

```python
from pybasin import BasinStabilityStudy

# Create adaptive sampling estimator
as_estimator = BasinStabilityStudy(
    system=system,
    initial_samples=100,
    max_samples=1000,
    uncertainty_threshold=0.1
)

# Estimate with adaptive sampling
as_results = as_estimator.estimate(bounds)

print(f"Samples used: {as_results.n_samples}")
print(f"Convergence achieved: {as_results.converged}")
```

## Custom Feature Extraction

You can define custom features for better classification:

```python
from pybasin import FeatureExtractor

class MyFeatureExtractor(FeatureExtractor):
    """Extract custom features from solutions."""

    def extract(self, solution):
        """Extract features from the solution."""
        t = solution.t
        y = solution.y

        features = {
            'final_x': y[0, -1],
            'final_y': y[1, -1],
            'max_distance': np.max(np.sqrt(y[0]**2 + y[1]**2)),
            'period': self._estimate_period(t, y),
        }
        return features

    def _estimate_period(self, t, y):
        """Estimate the period of oscillation."""
        # Your period estimation logic here
        return 0.0

# Use custom feature extractor
estimator = BasinStabilityEstimator(
    system=system,
    feature_extractor=MyFeatureExtractor()
)
```

## Working with High-Dimensional Systems

For systems with more than 2 dimensions:

```python
class LorenzSystem(ODESystem):
    """The Lorenz system."""

    def __init__(self, sigma=10, rho=28, beta=8/3):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def dynamics(self, t, state):
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz])

    def classify_attractor(self, solution):
        # Classification logic for Lorenz attractors
        final_state = solution.y[:, -1]
        if final_state[2] > 20:
            return 0
        return 1

# Sample in 3D space
lorenz = LorenzSystem()
estimator = BasinStabilityEstimator(lorenz)
bounds = [(-20, 20), (-30, 30), (0, 50)]
results = estimator.estimate(bounds)
```

## Saving and Loading Results

```python
# Save results
results.save('my_results.json')

# Load results
from pybasin import BasinStabilityResult
loaded_results = BasinStabilityResult.load('my_results.json')
```

## Next Steps

- Explore the [API Reference](../api/basin-stability-estimator.md)
- Check out the [Case Studies](../case-studies/overview.md) for real-world examples
