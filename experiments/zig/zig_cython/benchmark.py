"""Benchmark: Zig Dopri5 solver with Cython-compiled ODE.

Architecture:
    1. ODE is written in Python-like Cython syntax (pendulum_ode.pyx)
    2. Cython compiles it to native C code (no Python/GIL in hot loop)
    3. The C function pointer is passed to the Zig solver
    4. Zig runs the full integration in parallel across threads

The FFI boundary is crossed only ONCE (Python → Zig batch call).
Inside the hot loop, Zig calls the Cython-compiled C function directly.

Build steps before running:
    cd experiments/zig/zig_cython && zig build -Doptimize=ReleaseFast
    cd experiments/zig/zig_cython/odes && cythonize -i pendulum_ode.pyx
"""

import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer

N_SAMPLES = 10000
DIM = 2
T0 = 0.0
T1 = 1000.0
T_STEADY = 900.0
N_SAVE_TOTAL = 10000
FP_THRESHOLD = 0.01

# C function pointer type matching the Zig solver's COdeFn
ODE_FUNC_TYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
)

EXPERIMENT_DIR = Path(__file__).parent
ZIG_LIB_PATH = EXPERIMENT_DIR / "zig-out" / "lib" / "libzigsolve.so"
ODE_DIR = EXPERIMENT_DIR / "odes"


def load_zig_solver() -> ctypes.CDLL:
    """Load the Zig shared library and set up argument types."""
    if not ZIG_LIB_PATH.exists():
        msg = (
            f"Zig library not found at {ZIG_LIB_PATH}\n"
            "Build: cd experiments/zig_cffi && zig build -Doptimize=ReleaseFast"
        )
        raise FileNotFoundError(msg)

    lib = ctypes.CDLL(str(ZIG_LIB_PATH))

    # solve_ivp_batch_parallel — multi-threaded, for compiled ODE callbacks
    # First arg is a raw pointer address (usize), NOT a ctypes function wrapper.
    # This bypasses ctypes trampoline so Zig threads can call the C function directly.
    lib.solve_ivp_batch_parallel.argtypes = [
        ctypes.c_size_t,  # ode_fn_ptr (raw C function address)
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # params
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # ics
        ctypes.c_size_t,  # n_ics
        ctypes.c_size_t,  # dim
        ctypes.c_double,  # t0
        ctypes.c_double,  # t1
        ctypes.c_double,  # rtol
        ctypes.c_double,  # atol
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # t_eval
        ctypes.c_size_t,  # n_eval
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # result (output)
        ctypes.c_size_t,  # max_steps
        ctypes.c_size_t,  # n_threads (0 = auto)
    ]
    lib.solve_ivp_batch_parallel.restype = ctypes.c_int32

    return lib


def load_cython_ode() -> int:
    """Import the Cython-compiled ODE module and get its raw C function pointer.

    Returns the function address as a Python integer. This is passed directly
    to Zig as a usize, bypassing ctypes trampoline entirely.
    """
    sys.path.insert(0, str(ODE_DIR))
    try:
        import pendulum_ode  # type: ignore[import-not-found]
    except ImportError as exc:
        msg = (
            f"Cython ODE module not found in {ODE_DIR}\n"
            "Build: cd experiments/zig_cffi/odes && cythonize -i pendulum_ode.pyx"
        )
        raise ImportError(msg) from exc

    return pendulum_ode.get_fn_ptr()


def generate_initial_conditions(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
    seed: int = 42,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    rng = np.random.default_rng(seed)
    offset = np.arcsin(params[1] / params[2])
    theta_min = -np.pi + offset
    theta_max = np.pi + offset

    theta = rng.uniform(theta_min, theta_max, N_SAMPLES)
    theta_dot = rng.uniform(-10.0, 10.0, N_SAMPLES)

    return np.ascontiguousarray(np.column_stack([theta, theta_dot]))


def classify_trajectories(
    result: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    t_eval: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
    """Classify each trajectory as FP (0) or LC (1)."""
    steady_mask = t_eval >= T_STEADY
    steady_sol = result[:, steady_mask, 1]

    max_vals = np.max(steady_sol, axis=1)
    mean_vals = np.mean(steady_sol, axis=1)
    delta = np.abs(max_vals - mean_vals)

    return (delta >= FP_THRESHOLD).astype(np.int32)


def main() -> None:
    n_threads = os.cpu_count() or 4

    print("=" * 60)
    print("Zig + Cython Benchmark: Compiled ODE, Multi-threaded")
    print("=" * 60)

    lib = load_zig_solver()
    ode_fn = load_cython_ode()

    params = np.array([0.1, 0.5, 1.0], dtype=np.float64)
    t_eval = np.linspace(T0, T1, N_SAVE_TOTAL, dtype=np.float64)
    ics = generate_initial_conditions(params)

    print(f"\nParameters: alpha={params[0]}, T={params[1]}, K={params[2]}")
    print(f"N_SAMPLES={N_SAMPLES}, dim={DIM}, threads={n_threads}")
    print(f"Saving {N_SAVE_TOTAL} points from t={T0:.0f} to t={T1:.0f}")
    print("Solver: Zig Dopri5 (rtol=1e-8, atol=1e-6)")
    print("ODE: Cython-compiled C function (no GIL, no Python overhead)")

    # Pre-allocate output buffer
    result = np.empty(N_SAMPLES * N_SAVE_TOTAL * DIM, dtype=np.float64)

    # --- First run (cold) ---
    print("\n" + "-" * 60)
    print(f"First run: {N_SAMPLES} ICs on {n_threads} threads...")
    print("-" * 60)

    start = time.perf_counter()
    ret = lib.solve_ivp_batch_parallel(
        ode_fn,
        params,
        ics,
        N_SAMPLES,
        DIM,
        T0,
        T1,
        1e-8,
        1e-6,
        t_eval,
        N_SAVE_TOTAL,
        result,
        1_000_000,
        0,  # n_threads=0 → auto-detect
    )
    elapsed_first = time.perf_counter() - start

    if ret != 0:
        print(f"Solver failed with code {ret}")
        return

    elapsed_ms_first = elapsed_first * 1000
    us_per_ic_first = (elapsed_first * 1_000_000) / N_SAMPLES
    print(f"First run:  {elapsed_ms_first:.1f} ms ({us_per_ic_first:.1f} μs per IC)")

    # --- Second run (warm caches) ---
    print("\n" + "-" * 60)
    print("Second run (warm caches)...")
    print("-" * 60)

    start = time.perf_counter()
    ret = lib.solve_ivp_batch_parallel(
        ode_fn,
        params,
        ics,
        N_SAMPLES,
        DIM,
        T0,
        T1,
        1e-8,
        1e-6,
        t_eval,
        N_SAVE_TOTAL,
        result,
        1_000_000,
        0,
    )
    elapsed = time.perf_counter() - start

    if ret != 0:
        print(f"Solver failed with code {ret}")
        return

    elapsed_ms = elapsed * 1000
    us_per_ic = (elapsed * 1_000_000) / N_SAMPLES
    print(f"Second run: {elapsed_ms:.1f} ms ({us_per_ic:.1f} μs per IC)")

    # --- Classify ---
    trajectories = result.reshape(N_SAMPLES, N_SAVE_TOTAL, DIM)
    labels = classify_trajectories(trajectories, t_eval)

    fp_count = int(np.sum(labels == 0))
    lc_count = int(np.sum(labels == 1))
    fp_frac = fp_count / N_SAMPLES
    lc_frac = lc_count / N_SAMPLES

    print("\n" + "=" * 60)
    print("Basin Stability Results")
    print("=" * 60)
    print(f"  Fixed Point (FP): {fp_count:5} / {N_SAMPLES}  =  {fp_frac:.4f}")
    print(f"  Limit Cycle (LC): {lc_count:5} / {N_SAMPLES}  =  {lc_frac:.4f}")
    print("=" * 60)

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("Performance comparison (10,000 ICs, 10k save points)")
    print("=" * 60)
    print(f"  Zig native (ODE in Zig):       ~494    ms")
    print(f"  Zig + Cython ODE:              {elapsed_ms:>8.1f}  ms  ← this run")
    print(f"  Diffrax/JAX (JIT, CPU):        ~15238  ms")
    print(f"  Zig + Python ODE callback:     ~102539 ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
