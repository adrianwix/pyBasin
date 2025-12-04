With --comprehensive:


BATCH EXTRACTION RESULTS (CPU MODE)
--------------------------------------------------------------------------------
  JAX parallel warmup (783 features):          27806.89ms
  JAX parallel post-warmup:                    25535.51ms
  tsfresh (75 features):                        3092.98ms

without:

--------------------------------------------------------------------------------
BATCH EXTRACTION RESULTS (CPU MODE)
--------------------------------------------------------------------------------
  JAX parallel warmup (41 features):             710.15ms
  JAX parallel post-warmup:                       36.36ms
  tsfresh (41 features):                         311.81ms
--------------------------------------------------------------------------------
  JAX parallel speedup:                                8.6x

  Per-series timing:
    JAX parallel:   36.36us/series
    tsfresh:        311.81us/series

with --all

BATCH FEATURE EXTRACTION BENCHMARK
====================================================================================================

Device: TFRT_CPU_0
Data: 200 timesteps, 1000 batches x 1 states = 1000 series
Feature set: all (72 JAX features)
tsfresh n_jobs=24
Timing tsfresh bulk extraction...

Timing JAX bulk extraction (parallel)...


--------------------------------------------------------------------------------
BATCH EXTRACTION RESULTS (CPU MODE)
--------------------------------------------------------------------------------
  JAX parallel warmup (783 features):          28473.42ms
  JAX parallel post-warmup:                    25607.85ms
  tsfresh (72 features):                         818.87ms
--------------------------------------------------------------------------------
  JAX parallel speedup:                                0.0x

  Per-series timing:
    JAX parallel:   25607.85us/series
    tsfresh:        818.87us/series
