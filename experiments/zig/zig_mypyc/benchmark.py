"""Benchmark: Zig Dopri5 solver with mypyc-compiled ODE callback via ctypes.

Architecture:
    1. ODE math is written as a typed Python function (pendulum_ode.py)
    2. mypyc compiles it to a CPython C extension
    3. A ctypes callback wrapper bridges the C extension to the Zig solver
    4. Zig calls the wrapper on every RK stage evaluation (single-threaded)

Unlike Cython, mypyc cannot export a raw C function pointer with a custom
signature. The ODE is still called through the ctypes trampoline, so the
FFI overhead per call remains. Only the ODE math itself runs as compiled C.

Build steps before running:
    cd experiments/zig/zig_mypyc && zig build -Doptimize=ReleaseFast
    cd experiments/zig/zig_mypyc/odes && uv run mypyc pendulum_ode.py
"""

import ctypes
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
            "Build: cd experiments/zig/zig_mypyc && zig build -Doptimize=ReleaseFast"
        )
        raise FileNotFoundError(msg)

    lib = ctypes.CDLL(str(ZIG_LIB_PATH))

    lib.solve_ivp_batch.argtypes = [
        ODE_FUNC_TYPE,
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
    ]
    lib.solve_ivp_batch.restype = ctypes.c_int32

    return lib


def load_mypyc_ode() -> ctypes._CFuncPtr:  # type: ignore[type-arg]
    """Load the mypyc-compiled ODE and wrap it in a ctypes callback.

    The mypyc-compiled `compute()` does the math as native C.
    The ctypes wrapper handles pointer unpacking and is still interpreted Python.
    """
    sys.path.insert(0, str(ODE_DIR))
    try:
        from pendulum_ode import compute  # type: ignore[import-not-found]
    except ImportError as exc:
        msg = (
            f"mypyc ODE module not found in {ODE_DIR}\n"
            "Build: cd experiments/zig/zig_mypyc/odes && uv run mypyc pendulum_ode.py"
        )
        raise ImportError(msg) from exc

    @ODE_FUNC_TYPE
    def pendulum_ode_callback(
        t: float,
        y: ctypes.POINTER(ctypes.c_double),  # type: ignore[type-arg]
        dydt: ctypes.POINTER(ctypes.c_double),  # type: ignore[type-arg]
        params: ctypes.POINTER(ctypes.c_double),  # type: ignore[type-arg]
        dim: int,
    ) -> None:
        d0, d1 = compute(y[0], y[1], params[0], params[1], params[2])
        dydt[0] = d0
        dydt[1] = d1

    return pendulum_ode_callback


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

    return np.column_stack([theta, theta_dot])


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
    print("=" * 60)
    print("Zig + mypyc Benchmark: Compiled ODE, ctypes callback")
    print("=" * 60)

    lib = load_zig_solver()
    ode_callback = load_mypyc_ode()

    params = np.array([0.1, 0.5, 1.0], dtype=np.float64)
    t_eval = np.linspace(T0, T1, N_SAVE_TOTAL, dtype=np.float64)
    ics = generate_initial_conditions(params)

    print(f"\nParameters: alpha={params[0]}, T={params[1]}, K={params[2]}")
    print(f"N_SAMPLES={N_SAMPLES}, dim={DIM}")
    print(f"Saving {N_SAVE_TOTAL} points from t={T0:.0f} to t={T1:.0f}")
    print("Solver: Zig Dopri5 (rtol=1e-8, atol=1e-6)")
    print("ODE: mypyc-compiled compute() + ctypes callback wrapper")

    # --- Batch solve (100 ICs, extrapolate) ---
    n_batch = min(N_SAMPLES, 100)

    print("\n" + "-" * 60)
    print(f"Solving {n_batch} ICs (batch, single-threaded)...")
    print("-" * 60)

    batch_result = np.empty(n_batch * N_SAVE_TOTAL * DIM, dtype=np.float64)

    start = time.perf_counter()
    ret = lib.solve_ivp_batch(
        ode_callback,
        params,
        np.ascontiguousarray(ics[:n_batch]),
        n_batch,
        DIM,
        T0,
        T1,
        1e-8,
        1e-6,
        t_eval,
        N_SAVE_TOTAL,
        batch_result,
        1_000_000,
    )
    elapsed = time.perf_counter() - start

    if ret != 0:
        print(f"Solver failed with code {ret}")
        return

    elapsed_ms = elapsed * 1000
    us_per_ic = (elapsed * 1_000_000) / n_batch
    est_full_s = (elapsed / n_batch) * N_SAMPLES

    print(f"\n{n_batch} ICs done in {elapsed_ms:.1f} ms ({us_per_ic:.1f} us per IC)")
    print(f"Estimated time for {N_SAMPLES} ICs: {est_full_s:.1f} s")

    # --- Classify ---
    trajectories = batch_result.reshape(n_batch, N_SAVE_TOTAL, DIM)
    labels = classify_trajectories(trajectories, t_eval)

    fp_count = int(np.sum(labels == 0))
    lc_count = int(np.sum(labels == 1))
    fp_frac = fp_count / n_batch
    lc_frac = lc_count / n_batch

    print("\n" + "=" * 60)
    print(f"Basin Stability Results ({n_batch} ICs)")
    print("=" * 60)
    print(f"  Fixed Point (FP): {fp_count:5} / {n_batch}  =  {fp_frac:.4f}")
    print(f"  Limit Cycle (LC): {lc_count:5} / {n_batch}  =  {lc_frac:.4f}")
    print("=" * 60)

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("Performance comparison (estimated for 10,000 ICs)")
    print("=" * 60)
    print(f"  Zig native (24 threads):       ~494    ms")
    print(f"  Zig + Cython ODE (24 threads): ~591    ms")
    print(f"  Zig + mypyc ODE callback:      ~{est_full_s * 1000:.0f} ms  <- this run")
    print(f"  Zig + Python ODE callback:     ~105000 ms")
    print(f"  Diffrax/JAX (JIT, CPU):        ~15238  ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
