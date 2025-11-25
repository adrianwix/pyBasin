Here’s a compact summary you can use as an implementation guide for your new feature-extraction design.

---

## 1. Core design philosophy

* **Separate concerns**:

  * `FeatureExtractor` → turns trajectories into numeric feature vectors.
  * Classifier (e.g. KNN) → maps feature vectors to attractor labels / basins.
* **Goal**: Users should be able to build feature extractors for most basin-stability problems *without* writing system-specific one-off classes.

---

## 2. Contract of `FeatureExtractor`

* Base abstract class (Python-style):

  ```python
  class FeatureExtractor(ABC):
      def __init__(self, exclude_states=None, time_steady: float = 0.0):
          self.exclude_states = exclude_states
          self.time_steady = time_steady

      def filter_time(self, solution):
          # solution.t: (T,), solution.y: (T, B, S)
          # returns y_filtered: (T_after, B, S)
          ...

      def filter_states(self, y):
          # y: (T_after, B, S)
          # returns y_filtered: (T_after, B, S_filtered)
          ...

      @abstractmethod
      def extract_feat(self, solution) -> np.ndarray:
          """
          Input: solution with (t: (T,), y: (T, B, S))
          Output: features: (B, F)  # numeric feature matrix per trajectory
          """
          ...
  ```

* `compute_bs` only needs:

  ```python
  X = feature_extractor.extract_feat(solution)  # (B, F)
  y_pred = knn.predict(X)
  ```

  It never sees labels or one-hot encodings from the extractor.

---

## 3. Generic building blocks

### 3.1. StatsExtractor

* Purpose: generic statistics over time for selected states.

* Usage examples:

  * `mean`, `max`, `max_abs`, `std` per state over post-transient window.
  * Used to reconstruct logic like:

    * Pendulum: `mean_w`, `max_w`, then `delta = |max_w - mean_w|`
    * Lorenz: `mean(x)`
    * Friction: `max_abs` of a particular state

* Interface:

  ```python
  class StatsExtractor(FeatureExtractor):
      def __init__(self, state_indices=None, stats=("mean", "max", "max_abs", "std"),
                   exclude_states=None, time_steady: float = 0.0):
          ...

      def extract_feat(self, solution) -> np.ndarray:
          # returns (B, S_sel * len(stats)) flattened
          ...
  ```

### 3.2. BoundednessExtractor

* Purpose: detect diverging / unbounded trajectories via `max_abs` across time and states.
* Used to reproduce Lorenz’s “|y| > threshold ⇒ unbounded” logic.

  ```python
  class BoundednessExtractor(FeatureExtractor):
      def __init__(self, state_indices=None, exclude_states=None,
                   time_steady: float = 0.0):
          ...

      def extract_feat(self, solution) -> np.ndarray:
          # returns (B, 1) = max_abs over time & selected states
          ...
  ```

---

## 4. Composition and derived features

Since `compute_bs` is a black box, *all* feature logic must live inside the extractor objects you pass in.

### 4.1. CompositeExtractor

* Concatenates outputs of multiple extractors:

  ```python
  class CompositeExtractor(FeatureExtractor):
      def __init__(self, extractors: list[FeatureExtractor]):
          self.extractors = extractors

      def extract_feat(self, solution) -> np.ndarray:
          feats = [e.extract_feat(solution) for e in self.extractors]
          return np.concatenate(feats, axis=1)  # (B, sum F_i)
  ```

### 4.2. DerivedFeatureExtractor

* Wraps a “base” extractor and adds hand-crafted derived features (like `delta`):

  ```python
  class DerivedFeatureExtractor(FeatureExtractor):
      def __init__(self, base: FeatureExtractor, func):
          """
          func: (X_base: (B, F_base)) -> D: (B, F_new)
          """
          self.base = base
          self.func = func

      def extract_feat(self, solution) -> np.ndarray:
          X = self.base.extract_feat(solution)   # (B, F_base)
          D = self.func(X)                      # (B, F_new)
          return np.concatenate([X, D], axis=1) # (B, F_base + F_new)
  ```

* This is where case-specific “delta = |max - mean|” lives, without touching `compute_bs`.

---

## 5. Re-expressing the case studies with the new design

### 5.1. Pendulum

* Old: custom `PendulumFeatureExtractor` returning one-hot.
* New: generic extractor + derived feature:

  ```python
  base_pendulum = StatsExtractor(
      state_indices=[1],             # angular velocity
      stats=("mean", "max"),
      time_steady=0.9 * T_final
  )

  def pendulum_delta(X):
      mean_w = X[:, 0]
      max_w  = X[:, 1]
      delta  = np.abs(max_w - mean_w)
      return delta[:, None]         # (B, 1)

  pendulum_features = DerivedFeatureExtractor(base_pendulum, pendulum_delta)
  # Features: (B, 3) = [mean_w, max_w, delta]
  ```

### 5.2. Lorenz

```python
lorenz_features = CompositeExtractor([
    StatsExtractor(
        state_indices=[0],           # x
        stats=("mean",),
        time_steady=0.9 * T_final
    ),
    BoundednessExtractor(
        state_indices=None,
        time_steady=0.9 * T_final
    )
])
# Features: (B, 2) = [mean_x, max_abs_all_states]
```

### 5.3. Friction oscillator

```python
friction_features = StatsExtractor(
    state_indices=[1],
    stats=("max_abs",),
    time_steady=0.9 * T_final
)
# Features: (B, 1) = max_abs of second state
```

KNN (configured elsewhere) finds clusters corresponding to FP vs LC.

---

## 6. Training and classification flow

1. **Training phase (offline, user side)**:

   * Choose / build a `FeatureExtractor` (often `CompositeExtractor` + friends).
   * Compute `X_train = extractor.extract_feat(solutions_train)`.
   * Use known attractor labels to build `y_train`.
   * Fit `KNNClassifier` (or clustering) on `(X_train, y_train)`.

2. **Basin stability computation (`compute_bs`)**:

   * Integrate trajectories → `solution`.
   * `X = feature_extractor.extract_feat(solution)` (B, F).
   * `labels = knn.predict(X)`.
   * Convert to one-hot if needed.

---

## 7. Implementation to-do list

When you start coding the new feature-extractor type, you can follow this checklist:

1. Implement base `FeatureExtractor` with:

   * `filter_time(solution)`
   * `filter_states(y)`
   * abstract `extract_feat(solution) -> (B, F)`
2. Implement generic building blocks:

   * `StatsExtractor`
   * `BoundednessExtractor`
3. Implement composition utilities:

   * `CompositeExtractor`
   * `DerivedFeatureExtractor`
4. Replace per-case `*FeatureExtractor` classes with:

   * Small “assembly scripts” that build instances of the above for each system.
5. Ensure `compute_bs` uses only:

   * `X = extractor.extract_feat(solution)`
   * then the KNN / clustering logic.

You can use this summary as a blueprint for refactoring your current implementation toward a more generic, reusable feature-extraction layer.
