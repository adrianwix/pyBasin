**Features And Methods (Merged)**

This document lists, in a single merged view, the computations and rule-based methods currently used by the feature extractors in `case_studies/*/*_feature_extractor.py`. It shows the mathematical operation, where it appears, and notes about how the result is currently used (often thresholded into one-hot encodings). It also gives short recommendations for returning continuous feature vectors appropriate for downstream clustering.

**Common Preprocessing**: All extractors call `FeatureExtractor.filter_time(solution)` to remove transient behavior before computing features. If `solution.time` is the time vector and `t0` is the `time_steady` threshold, the filtered trajectory is

$$y_{f}(t, i, s) = y(t,i,s) \quad \text{for } t>t_0$$

where indices are: time steps $t$, batch / trajectory $i$, and state variable $s$.

- **Data type & shapes**: all computations use PyTorch tensors with shape `(N, B, S)` for `solution.y` and yield features of shape `(B, F)`.

**Cross-reference**: A comprehensive catalog of feature families and expanded extraction ideas is maintained in `pyBasinWorkspace/FEATURE_EXTRACTOR_LIST.md`. That file is the canonical reference for implementing additional time-domain, frequency-domain, recurrence, dimensionality, and topological features. The optional spectral/temporal features listed below (`dominant_freq_s`, `spectral_entropy_s`, `autocorr_time_s`) are described in more detail in the Frequency-domain (section 3) and Time-domain (section 2) parts of `FEATURE_EXTRACTOR_LIST.md`.

**Merged list of computations (explicit formulas + where used)**

- **Maximum (over time)**:
  - Formula: $\max_t\, x(t)$ for a scalar time series $x(t)$.
  - How used: `duffing_feature_extractor` computes $\max_t$ of the first state (displacement). `pendulum_feature_extractor` uses `angular_velocity.max(dim=0)` as part of a delta calculation.
  - Current use: returned as part of the feature vector (Duffing) or used inside a rule (Pendulum).
  - Recommendation: return as continuous feature `max(x)`.

- **Minimum / Peak-to-peak**:
  - Not computed directly in the present code, but `max` and `min` are natural complements.
  - Recommendation: provide `min(x)` and `ptp(x)=max(x)-min(x)` for clustering.

- **Maximum absolute value**:
  - Formula: $\max_t |x(t)|$.
  - How used: `friction_feature_extractor` computes $\max_t |y_{:,1}|$ (second state) to detect amplitude. `lorenz_feature_extractor` computes $\max_t |y|$ across states to detect unbounded trajectories (by checking if any state exceeds a large threshold).
  - Current use: thresholded (`<=0.2` → FP, `>0.2` → LC) in `friction`; threshold `>195` marks unbounded in `lorenz`.
  - Recommendation: return `max_abs(x)` as continuous; optionally include a separate boolean `unbounded_flag` for filtering out divergent trajectories.

- **Mean (time average)**:
  - Formula: $\bar{x} = \frac{1}{N}\sum_t x(t)$.
  - How used: `lorenz_feature_extractor` computes mean of the first state $x$ and uses its sign ($\bar{x} >0$) to classify regimes. Many extractors implicitly use the mean in other statistics (e.g., std, delta uses mean).
  - Current use: sign test (positive vs negative) in `lorenz`.
  - Recommendation: return `mean(x)` (continuous) and also normalized mean (e.g., mean divided by std) if scale invariance is desired.

- **Standard deviation**:
  - Formula: $\mathrm{std}(x) = \sqrt{\frac{1}{N-1} \sum_t (x(t)-\bar{x})^2}$ (PyTorch default unbiased sample std used in code).
  - How used: `duffing_feature_extractor` computes std for the first state and returns it as a feature.
  - Recommendation: keep `std(x)` as a continuous feature; also consider `var(x)` or normalized measures (e.g., coefficient of variation `std/mean`).

- **Delta = max − mean**:
  - Formula: $\Delta(x)=\max_t x(t) - \bar{x}$.
  - How used: `pendulum_feature_extractor` computes `delta = max(angular_velocity) - mean(angular_velocity)` and thresholds `delta < 0.01` to detect fixed points vs limit cycles.
  - Current use: boolean threshold → one‑hot.
  - Recommendation: return `delta(x)` as a continuous feature (or both `max` and `mean`) so clustering can use the continuous separation.

- **Difference & absolute deviation checks**:
  - Generic operations used: absolute value `|x|`, comparisons `>`, `<=`, boolean masking.
  - How used: thresholds (e.g., `max_abs <= 0.2`, `delta < 0.01`, `mean(x) > 0`, `max_abs > 195`).
  - Recommendation: rather than mapping immediately to one-hot, include the raw scalar(s) used in thresholds in the feature vector (e.g., `max_abs`, `delta`, `mean`) and optionally keep the boolean threshold outputs as extra features.

**Current one-hot encodings (where they come from)**

- `friction_feature_extractor.py`: two classes `FP` and `LC` encoded as `[1,0]` and `[0,1]`. Decision rule: if $\max_t |y_2(t)| \le 0.2$ then `FP` else `LC`.
- `lorenz_feature_extractor.py`: two regimes `S1` (mean x > 0) and `S2` (mean x < 0) encoded as `[1,0]` and `[0,1]`; any trajectory found to have a state magnitude greater than `195` is treated as unbounded and encoded as `[0,0]`.
- `pendulum_feature_extractor.py`: `FP` vs `LC` by `\Delta(\dot{\theta}) < 0.01`; one-hot as above.
- `duffing_feature_extractor.py`: not one-hot — returns continuous features `[max, std]` for the primary (first) state variable.

**Interpretation & intent**

- The current code implements deterministic, hand-tuned summary-statistics plus simple rule-based thresholds to produce discrete labels (one-hot). That is appropriate when the set of attractor types is known and separable by simple statistics.
- For clustering (unsupervised) you generally want continuous, informative features as input. The one-hot outputs throw away intra-class structure and scale information. Instead, return the underlying scalar summary statistics and let the clustering algorithm decide grouping boundaries.

**Suggested canonical feature vector to return (per trajectory)**

Include the following scalars computed on steady-state portion `y_f` for each relevant state variable or for selected states:

- `mean_s` = $\bar{y}_s$ (time mean of state s)
- `std_s` = $\mathrm{std}(y_s)$ (time std)
- `max_s` = $\max_t y_s(t)$
- `min_s` = $\min_t y_s(t)$
- `max_abs_s` = $\max_t |y_s(t)|`
- `ptp_s` = `max_s - min_s` (peak-to-peak)
- `rms_s` = $\sqrt{\frac{1}{N} \sum_t y_s(t)^2}$ (root-mean-square)
- `delta_s` = `max_s - mean_s` (already used for pendulum)

Optional spectral / temporal features (helpful for oscillatory systems):

- `dominant_freq_s` = frequency of largest FFT peak for state s
- `spectral_entropy_s` = spectral entropy across power spectral density
- `autocorr_time_s` = first zero crossing / decay time of autocorrelation

Notes and cross-references for these optional spectral/temporal features:

- `dominant_freq_s`
  - Where: `FEATURE_EXTRACTOR_LIST.md` → section "3. Frequency-domain features (FFT-based)".
  - Description match: "Dominant frequency index or value: Location of maximal spectral peak (excluding DC)."
  - Use: compute the power spectral density (PSD) of the scalar time series `s(t)` (e.g., via FFT), find the frequency bin with maximum power (excluding the zero-frequency/DC bin), and return that frequency (or its index) as `dominant_freq_s`.

- `spectral_entropy_s`
  - Where: `FEATURE_EXTRACTOR_LIST.md` → section "3. Frequency-domain features (FFT-based)".
  - Description match: "Spectral entropy: −Σ p_i log p_i with p_i = P_i / Σ P_i."
  - Use: normalize the PSD into a probability distribution `p_i` over frequency bins and compute `-sum(p_i * log(p_i))`. Low values indicate concentrated spectra (clean periodic); high values indicate broad / chaotic spectra.

- `autocorr_time_s`
  - Where: `FEATURE_EXTRACTOR_LIST.md` → section "2. Time-domain features (on one or few coordinates)" (that section lists autocorrelation values at small lags).
  - Note: `autocorr_time_s` in this document is a more specific single-value summary (for example, the first zero-crossing of the autocorrelation or the 1/e decay time). The `FEATURE_EXTRACTOR_LIST.md` suggests computing autocorrelation at relevant lags — both are consistent and complementary. You can compute multiple lag autocorrelations (per the list) and reduce them to a single timescale summary for `autocorr_time_s`.

Notes about scaling: include feature normalization (per-feature z-score or robust scaling) before clustering to avoid scale-driven clusters.

**Minimal migration plan (code changes) to support clustering**

1. Modify case-specific extractors to return continuous feature vectors instead of mapping directly to one-hot. Concretely:
   - Replace `out[...] = fp_tensor` / `lc_tensor` rules with `out[i] = torch.tensor([max_abs, mean, std, delta, ...])`.
2. Keep current one-hot outputs as an optional wrapper `classify_from_features(features, thresholds)` for backwards compatibility.
3. Add a generic `StatFeatureExtractor` that takes a config list (e.g., `['mean', 'std', 'max_abs']`) and computes them for any system.

**References to code locations**

- `pyBasinWorkspace/case_studies/duffing_oscillator/duffing_feature_extractor.py` — computes `max` and `std` of state 0.
- `pyBasinWorkspace/case_studies/friction/friction_feature_extractor.py` — computes `max_abs` of state 1 and thresholds `<=0.2`.
- `pyBasinWorkspace/case_studies/lorenz/lorenz_feature_extractor.py` — computes per-state `max_abs` to detect unbounded (`>195`) and `mean` of state 0 for sign-based classification.
- `pyBasinWorkspace/case_studies/pendulum/pendulum_feature_extractor.py` — computes `delta = max - mean` of angular velocity and thresholds `<0.01`.

**Closing notes**

All current extractors are short, interpretable, and time-domain based. For clustering, prefer returning the continuous scalar summaries (the underlying statistics used in the rules) and augment them with basic spectral and scale-invariant features where useful. If you want, I can implement a `StatFeatureExtractor` that computes a configurable set of features and update the case study extractors to use it, preserving the current one-hot classification as an optional post-processing step.
