"""Zig ODE Solver Python Interface.

This module provides a Python interface to the high-performance Zig ODE solver.

Architecture:
- **Solver library** (libzig_ode_solver.so): Pre-compiled Dopri5 solver that accepts
  a function pointer to any ODE. Built once with `zig build`.
- **ODE libraries** (lib<name>.so): User-defined ODEs compiled on-demand. Each ODE
  is a single .zig file that exports `ode_func`, `ode_dim`, `ode_param_size`.

Usage:
    from zig_solver import ZigODESolver

    solver = ZigODESolver()

    # Solve pendulum ODE (compiles pendulum.zig on first use if needed)
    t, y = solver.solve(
        ode_name="pendulum",
        y0=[0.5, 0.0],
        t_span=(0.0, 10.0),
        t_eval=np.linspace(0, 10, 100),
        params={"alpha": 0.1, "T": 0.5, "K": 1.0},
    )

Adding new ODEs:
    1. Create user_odes/your_ode.zig following the template (see pendulum.zig)
    2. Define ctypes Params struct in ODE_PARAM_TYPES below
    3. Call solver.solve("your_ode", ...) - it auto-compiles on first use
"""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# ============================================================
# Path Configuration
# ============================================================

ZIG_ROOT = Path(__file__).parent
SRC_DIR = ZIG_ROOT / "src"
USER_ODE_DIR = ZIG_ROOT / "user_odes"
LIB_DIR = ZIG_ROOT / "zig-out" / "lib"
ODE_LIB_DIR = ZIG_ROOT / "zig-out" / "ode_libs"
SOLVER_LIB_NAME = "libzig_ode_solver.so"
SOLVER_HASH_FILE = ZIG_ROOT / ".solver_hash"


# ============================================================
# ODE Parameter Structures (ctypes)
# ============================================================
# Each ODE needs a ctypes Structure that matches the Zig struct layout.
# The field order and types MUST match exactly (extern struct in Zig).


class PendulumParams(ctypes.Structure):
    """Parameters for the pendulum ODE (must match Zig Params struct)."""

    _fields_ = [
        ("alpha", ctypes.c_double),
        ("T", ctypes.c_double),
        ("K", ctypes.c_double),
    ]


# Registry mapping ODE names to their ctypes param struct
ODE_PARAM_TYPES: dict[str, type[ctypes.Structure]] = {
    "pendulum": PendulumParams,
}


# ============================================================
# Solver Library Management
# ============================================================


def _compute_solver_hash() -> str:
    """Compute hash of solver source files."""
    hasher = hashlib.sha256()
    for zig_file in sorted(SRC_DIR.rglob("*.zig")):
        hasher.update(zig_file.name.encode())
        hasher.update(zig_file.read_bytes())
    build_zig = ZIG_ROOT / "build.zig"
    if build_zig.exists():
        hasher.update(build_zig.read_bytes())
    return hasher.hexdigest()


def _solver_needs_rebuild() -> bool:
    """Check if solver library needs rebuilding."""
    lib_path = LIB_DIR / SOLVER_LIB_NAME
    if not lib_path.exists():
        return True
    if not SOLVER_HASH_FILE.exists():
        return True
    return _compute_solver_hash() != SOLVER_HASH_FILE.read_text().strip()


def rebuild_solver(optimize: bool = True) -> None:
    """Rebuild the solver library."""
    cmd = ["zig", "build"]
    if optimize:
        cmd.extend(["-Doptimize=ReleaseFast"])

    print(f"Building solver library: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=ZIG_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Solver build failed:\n{result.stderr}")

    SOLVER_HASH_FILE.write_text(_compute_solver_hash())
    print("Solver build successful!", file=sys.stderr)


# ============================================================
# ODE Library Management
# ============================================================


def _get_ode_source(ode_name: str) -> Path:
    """Get path to ODE source file."""
    return USER_ODE_DIR / f"{ode_name}.zig"


def _get_ode_lib(ode_name: str) -> Path:
    """Get path to compiled ODE library."""
    return ODE_LIB_DIR / f"lib{ode_name}.so"


def _get_ode_hash_file(ode_name: str) -> Path:
    """Get path to ODE hash file."""
    return ODE_LIB_DIR / f".{ode_name}_hash"


def _compute_ode_hash(ode_name: str) -> str:
    """Compute hash of a single ODE source file."""
    source = _get_ode_source(ode_name)
    if not source.exists():
        raise FileNotFoundError(f"ODE source not found: {source}")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def ode_needs_compile(ode_name: str) -> bool:
    """Check if an ODE needs to be compiled.

    Returns True if:
    - The ODE library doesn't exist
    - The source file has changed since last compile
    """
    lib_path = _get_ode_lib(ode_name)
    hash_file = _get_ode_hash_file(ode_name)

    if not lib_path.exists():
        return True
    if not hash_file.exists():
        return True
    return _compute_ode_hash(ode_name) != hash_file.read_text().strip()


WRAPPER_TEMPLATE = SRC_DIR / "ode_wrapper_template.zig"


def _get_wrapper_path(ode_name: str) -> Path:
    """Get path to generated wrapper file."""
    return ODE_LIB_DIR / f"_wrapper_{ode_name}.zig"


def compile_ode(ode_name: str, optimize: bool = True) -> None:
    """Compile a single ODE to a shared library.

    This generates a wrapper that imports the user's ODE file and exports
    the C ABI functions. The user's ODE file only needs to define:
    - DIM: State dimension
    - Params: Parameter struct (extern struct)
    - ode: The ODE function with clean Zig signature

    :param ode_name: Name of the ODE (without .zig extension).
    :param optimize: If True, use ReleaseFast optimization.
    """
    source = _get_ode_source(ode_name)
    if not source.exists():
        raise FileNotFoundError(
            f"ODE source not found: {source}\n"
            f"Create {source} following the template in user_odes/pendulum.zig"
        )

    # Ensure output directory exists
    ODE_LIB_DIR.mkdir(parents=True, exist_ok=True)

    # Generate wrapper file in ZIG_ROOT (not in src/) to allow importing user_odes/
    # Zig forbids imports from parent directories relative to the source file
    wrapper_path = ZIG_ROOT / f"_wrapper_{ode_name}.zig"
    template = WRAPPER_TEMPLATE.read_text()

    # Path from ZIG_ROOT to user_odes/
    relative_path = f"user_odes/{ode_name}.zig"
    wrapper_code = template.format(user_ode_path=relative_path)
    wrapper_path.write_text(wrapper_code)

    lib_path = _get_ode_lib(ode_name)
    opt_flag = "-OReleaseFast" if optimize else "-ODebug"

    # Use relative paths since we run from ZIG_ROOT
    wrapper_rel = wrapper_path.relative_to(ZIG_ROOT)
    lib_rel = lib_path.relative_to(ZIG_ROOT)

    cmd = [
        "zig",
        "build-lib",
        "-dynamic",
        opt_flag,
        str(wrapper_rel),
        f"-femit-bin={lib_rel}",
    ]

    print(f"Compiling ODE '{ode_name}': {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=ZIG_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        # Clean up wrapper on failure
        wrapper_path.unlink(missing_ok=True)
        raise RuntimeError(f"ODE compile failed:\n{result.stderr}")

    # Clean up wrapper (optional, keep for debugging)
    # wrapper_path.unlink(missing_ok=True)

    # Save hash
    _get_ode_hash_file(ode_name).write_text(_compute_ode_hash(ode_name))
    print(f"ODE '{ode_name}' compiled successfully!", file=sys.stderr)


def compile_ode_if_needed(ode_name: str, optimize: bool = True) -> bool:
    """Compile ODE if source has changed.

    :return: True if compilation was performed.
    """
    if ode_needs_compile(ode_name):
        compile_ode(ode_name, optimize)
        return True
    return False


def list_available_odes() -> list[str]:
    """List all ODE source files in user_odes/."""
    if not USER_ODE_DIR.exists():
        return []
    return [f.stem for f in USER_ODE_DIR.glob("*.zig")]


# ============================================================
# ODE Function Pointer Type
# ============================================================

# C function pointer type for ODE: void (*)(double t, double* y, double* dydt, void* params, size_t dim)
ODE_FUNC_TYPE = ctypes.CFUNCTYPE(
    None,  # return type (void)
    ctypes.c_double,  # t
    ctypes.POINTER(ctypes.c_double),  # y
    ctypes.POINTER(ctypes.c_double),  # dydt
    ctypes.c_void_p,  # params
    ctypes.c_size_t,  # dim
)


# ============================================================
# Loaded ODE Cache
# ============================================================


class LoadedODE:
    """Cached loaded ODE library with function pointer."""

    def __init__(self, name: str, lib: ctypes.CDLL) -> None:
        self.name = name
        self._lib = lib

        # Get function pointers
        self._lib.ode_func.argtypes = [
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._lib.ode_func.restype = None

        self._lib.ode_dim.argtypes = []
        self._lib.ode_dim.restype = ctypes.c_size_t

        self._lib.ode_param_size.argtypes = []
        self._lib.ode_param_size.restype = ctypes.c_size_t

    @property
    def dim(self) -> int:
        return self._lib.ode_dim()

    @property
    def param_size(self) -> int:
        return self._lib.ode_param_size()

    @property
    def func_ptr(self) -> ctypes.c_void_p:
        """Get raw function pointer to pass to solver."""
        return ctypes.cast(self._lib.ode_func, ctypes.c_void_p)


# ============================================================
# Solver Class
# ============================================================


class ZigODESolver:
    """Python interface to the Zig ODE solver.

    The solver library is loaded once. ODE libraries are compiled and loaded
    on-demand when first used.
    """

    def __init__(self, auto_rebuild_solver: bool = True) -> None:
        """Initialize the solver.

        :param auto_rebuild_solver: Rebuild solver library if sources changed.
        """
        if auto_rebuild_solver and _solver_needs_rebuild():
            rebuild_solver()

        lib_path = LIB_DIR / SOLVER_LIB_NAME
        if not lib_path.exists():
            raise FileNotFoundError(
                f"Solver library not found: {lib_path}\nRun rebuild_solver() to build it."
            )

        self._solver_lib = ctypes.CDLL(str(lib_path))
        self._setup_solver_functions()

        # Cache of loaded ODE libraries
        self._ode_cache: dict[str, LoadedODE] = {}

    def _setup_solver_functions(self) -> None:
        """Set up ctypes signatures for solver library."""
        # solve_ode(ode_fn, y0, dim, t0, t1, save_at, n_save, rtol, atol, max_steps, params, result)
        self._solver_lib.solve_ode.argtypes = [
            ctypes.c_void_p,  # ode_fn (function pointer)
            ctypes.POINTER(ctypes.c_double),  # y0
            ctypes.c_size_t,  # dim
            ctypes.c_double,  # t0
            ctypes.c_double,  # t1
            ctypes.POINTER(ctypes.c_double),  # save_at
            ctypes.c_size_t,  # n_save
            ctypes.c_double,  # rtol
            ctypes.c_double,  # atol
            ctypes.c_size_t,  # max_steps
            ctypes.c_void_p,  # params
            ctypes.POINTER(ctypes.c_double),  # result
        ]
        self._solver_lib.solve_ode.restype = ctypes.c_int32

        # solve_batch(ode_fn, y0s, n_ics, dim, t0, t1, save_at, n_save, rtol, atol, max_steps, params, results, n_threads)
        self._solver_lib.solve_batch.argtypes = [
            ctypes.c_void_p,  # ode_fn (function pointer)
            ctypes.POINTER(ctypes.c_double),  # y0s (n_ics * dim)
            ctypes.c_size_t,  # n_ics
            ctypes.c_size_t,  # dim
            ctypes.c_double,  # t0
            ctypes.c_double,  # t1
            ctypes.POINTER(ctypes.c_double),  # save_at
            ctypes.c_size_t,  # n_save
            ctypes.c_double,  # rtol
            ctypes.c_double,  # atol
            ctypes.c_size_t,  # max_steps
            ctypes.c_void_p,  # params
            ctypes.POINTER(ctypes.c_double),  # results (n_ics * n_save * dim)
            ctypes.c_size_t,  # n_threads
        ]
        self._solver_lib.solve_batch.restype = ctypes.c_int32

    def _load_ode(self, ode_name: str, auto_compile: bool = True) -> LoadedODE:
        """Load an ODE library, compiling if needed."""
        # Check cache first
        if ode_name in self._ode_cache:
            # Check if source changed
            if not ode_needs_compile(ode_name):
                return self._ode_cache[ode_name]
            # Source changed, need to reload
            del self._ode_cache[ode_name]

        # Compile if needed
        if auto_compile:
            compile_ode_if_needed(ode_name)

        lib_path = _get_ode_lib(ode_name)
        if not lib_path.exists():
            raise FileNotFoundError(
                f"ODE library not found: {lib_path}\nRun compile_ode('{ode_name}') to build it."
            )

        # Load library
        ode_lib = ctypes.CDLL(str(lib_path))
        loaded = LoadedODE(ode_name, ode_lib)
        self._ode_cache[ode_name] = loaded
        return loaded

    def get_dim(self, ode_name: str) -> int:
        """Get the dimension of an ODE's state space."""
        return self._load_ode(ode_name).dim

    def solve(
        self,
        ode_name: str,
        y0: NDArray[np.float64] | list[float],
        t_span: tuple[float, float],
        t_eval: NDArray[np.float64] | list[float],
        params: dict[str, float] | None = None,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        max_steps: int = 1_000_000,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Solve an ODE system.

        :param ode_name: Name of the ODE (e.g., "pendulum").
        :param y0: Initial conditions.
        :param t_span: Integration interval (t0, t1).
        :param t_eval: Times at which to store the solution.
        :param params: ODE parameters as a dictionary.
        :param rtol: Relative tolerance.
        :param atol: Absolute tolerance.
        :param max_steps: Maximum number of integration steps.
        :return: Tuple of (t_eval, solution) where solution has shape (len(t_eval), dim).
        """
        # Load ODE (compiles if needed)
        ode = self._load_ode(ode_name)

        # Get dimension
        dim = ode.dim

        # Convert inputs to numpy arrays
        y0_arr = np.asarray(y0, dtype=np.float64)
        t_eval_arr = np.asarray(t_eval, dtype=np.float64)

        if len(y0_arr) != dim:
            raise ValueError(f"y0 has length {len(y0_arr)}, expected {dim} for ODE '{ode_name}'")

        # Create params struct
        params_ptr = None
        if ode_name in ODE_PARAM_TYPES:
            param_type = ODE_PARAM_TYPES[ode_name]
            params_struct = param_type() if params is None else param_type(**params)
            params_ptr = ctypes.cast(ctypes.pointer(params_struct), ctypes.c_void_p)

        # Allocate output buffer
        n_save = len(t_eval_arr)
        result = np.zeros((n_save, dim), dtype=np.float64)

        # Call solver with ODE function pointer
        t0, t1 = t_span
        error_code = self._solver_lib.solve_ode(
            ode.func_ptr,
            y0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dim,
            t0,
            t1,
            t_eval_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_save,
            rtol,
            atol,
            max_steps,
            params_ptr,
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        # Check for errors
        if error_code != 0:
            error_messages = {
                -1: "Invalid ODE function pointer",
                -2: "Memory allocation failed",
                -3: "Solver failed",
                -4: "Thread spawn failed",
            }
            msg = error_messages.get(error_code, f"Unknown error {error_code}")
            raise RuntimeError(f"Zig solver error: {msg}")

        return t_eval_arr, result

    def solve_batch(
        self,
        ode_name: str,
        y0s: NDArray[np.float64],
        t_span: tuple[float, float],
        t_eval: NDArray[np.float64] | list[float],
        params: dict[str, float] | None = None,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        max_steps: int = 1_000_000,
        n_threads: int = 0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Solve an ODE system for multiple initial conditions in parallel.

        :param ode_name: Name of the ODE (e.g., "pendulum").
        :param y0s: Initial conditions array of shape (n_ics, dim).
        :param t_span: Integration interval (t0, t1).
        :param t_eval: Times at which to store the solution.
        :param params: ODE parameters as a dictionary.
        :param rtol: Relative tolerance.
        :param atol: Absolute tolerance.
        :param max_steps: Maximum number of integration steps.
        :param n_threads: Number of threads (0 = auto-detect CPU count).
        :return: Tuple of (t_eval, solutions) where solutions has shape (n_ics, len(t_eval), dim).
        """
        # Load ODE (compiles if needed)
        ode = self._load_ode(ode_name)
        dim = ode.dim

        # Convert inputs to numpy arrays
        y0s_arr = np.ascontiguousarray(y0s, dtype=np.float64)
        t_eval_arr = np.asarray(t_eval, dtype=np.float64)

        if y0s_arr.ndim != 2 or y0s_arr.shape[1] != dim:
            raise ValueError(f"y0s must have shape (n_ics, {dim}), got {y0s_arr.shape}")

        n_ics = y0s_arr.shape[0]
        n_save = len(t_eval_arr)

        # Create params struct
        params_ptr = None
        if ode_name in ODE_PARAM_TYPES:
            param_type = ODE_PARAM_TYPES[ode_name]
            params_struct = param_type() if params is None else param_type(**params)
            params_ptr = ctypes.cast(ctypes.pointer(params_struct), ctypes.c_void_p)

        # Allocate output buffer (n_ics * n_save * dim)
        results = np.zeros((n_ics, n_save, dim), dtype=np.float64)

        # Call solver with ODE function pointer
        t0, t1 = t_span
        error_code = self._solver_lib.solve_batch(
            ode.func_ptr,
            y0s_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_ics,
            dim,
            t0,
            t1,
            t_eval_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_save,
            rtol,
            atol,
            max_steps,
            params_ptr,
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_threads,
        )

        # Check for errors
        if error_code != 0:
            error_messages = {
                -1: "Invalid ODE function pointer",
                -2: "Memory allocation failed",
                -3: "Solver failed",
                -4: "Thread spawn failed",
            }
            msg = error_messages.get(error_code, f"Unknown error {error_code}")
            raise RuntimeError(f"Zig solver error: {msg}")

        return t_eval_arr, results


# ============================================================
# Main (for testing)
# ============================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Zig ODE Solver - Dynamic ODE Loading Test")
    print("=" * 60)

    # Show available ODEs
    print(f"\nAvailable ODE sources: {list_available_odes()}")

    # Force rebuild solver
    rebuild_solver(optimize=True)

    # Create solver
    solver = ZigODESolver(auto_rebuild_solver=False)

    # Test solve - this compiles pendulum.zig on first use
    print(f"\nPendulum dimension: {solver.get_dim('pendulum')}")

    t_eval = np.linspace(0, 10, 1000)
    params = {"alpha": 0.1, "T": 0.5, "K": 1.0}

    # Warm-up (includes first compile if needed)
    t, y = solver.solve("pendulum", [0.5, 0.0], (0.0, 10.0), t_eval, params)

    # Benchmark
    n_runs = 100
    start = time.perf_counter()
    for _ in range(n_runs):
        t, y = solver.solve("pendulum", [0.5, 0.0], (0.0, 10.0), t_eval, params)
    elapsed = time.perf_counter() - start

    print(f"\nBenchmark ({n_runs} runs):")
    print(f"  Total: {elapsed * 1000:.1f} ms")
    print(f"  Per solve: {elapsed / n_runs * 1000:.3f} ms")
    print(f"  Solution shape: {y.shape}")
    print(f"  Final state: θ={y[-1, 0]:.4f}, θ̇={y[-1, 1]:.4f}")
