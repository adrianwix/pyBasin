# Zig Solver Experiments

Benchmarks comparing different approaches to calling a Dopri5 (Dormand-Prince 5th order)
ODE solver for basin stability computation on the damped pendulum:
d0/dt = 0', d0'/dt = -a*0' + T - K*sin(0).

All variants solve 10,000 initial conditions, each integrated from t=0 to t=1000
with 10,000 save points. The solver uses adaptive step-size control (rtol=1e-8, atol=1e-6)
and dense output interpolation (Shampine's formula).

## Results

Measured on a 24-thread machine (AMD Ryzen 9 5900X). Each timing is the median of 3 runs.

| Approach                  | Time (ms) | us/IC   | Relative | Threads        |
| ------------------------- | --------- | ------- | -------- | -------------- |
| **Zig native**            | ~494      | ~49     | 1.0x     | 24             |
| **Zig + Cython ODE**      | ~591      | ~59     | 1.2x     | 24             |
| **Diffrax/JAX**           | ~15,238   | ~1,524  | 31x      | 1 (vectorized) |
| **Zig + mypyc ODE**       | ~108,000  | ~10,800 | 219x     | 1              |
| **Zig + Python callback** | ~105,000  | ~10,500 | 213x     | 1              |

The Diffrax benchmark lives separately under `experiments/diffrax/`.

## Directory Structure

```
zig/
  zig_native/             Pure Zig: ODE + solver + threading
  zig_python_callback/    Zig solver, ODE defined in Python (ctypes callback)
  zig_mypyc/              Zig solver, ODE compiled via mypyc (ctypes callback)
  zig_cython/             Zig solver, ODE compiled via Cython to C
```

### zig_native/

Everything in Zig. The ODE function is known at comptime, so the compiler inlines
it into the solver loop -- no function pointer indirection. Each thread reuses an
arena allocator with `.retain_capacity` across ICs.

```
zig_native/
  build.zig
  src/
    main.zig          Entry point, threading, basin stability classification
    dopri5.zig        Generic Dopri5 integrator (comptime-parameterized)
    pendulum.zig      ODE definition + parameter struct
```

**Build and run:**

```bash
cd experiments/zig/zig_native
zig build -Doptimize=ReleaseFast
./zig-out/bin/pendulum_solver
```

### zig_python_callback/

The Zig solver is compiled as a shared library (`.so`). Python defines the ODE,
wraps it as a `ctypes.CFUNCTYPE` callback, and passes it to the Zig solver.

Every RK stage evaluation crosses the Python<->C FFI boundary, which costs ~1-10 us
per call. With ~7 evaluations per step and thousands of steps per IC, this overhead
dominates. Single-threaded only, since the Python callback holds the GIL.

```
zig_python_callback/
  build.zig
  benchmark.py        Benchmark script (runs 100 ICs, extrapolates to 10k)
  src/
    solver_lib.zig    Dopri5 solver as shared library with C ABI exports
```

**Build and run:**

```bash
cd experiments/zig/zig_python_callback
zig build -Doptimize=ReleaseFast
uv run python benchmark.py
```

### zig_cython/

Same Zig solver library, but the ODE is written in Cython (`.pyx`) and compiled to
a native C function. Python obtains the raw C function pointer via `get_fn_ptr()` and
passes it to Zig as an integer address (`usize`). Zig casts it back to a function
pointer and calls it directly from spawned threads -- no GIL, no ctypes trampoline.

The `build.zig` sets `link_libc = true` so that `std.Thread.spawn` uses `pthread_create`
rather than raw `clone`. Without this, Zig-spawned threads lack glibc TLS initialization
and crash when the Cython code calls into libm (e.g., `sin()`).

```
zig_cython/
  build.zig
  benchmark.py        Benchmark script (10k ICs, multi-threaded)
  src/
    solver_lib.zig    Same Dopri5 solver, with solve_ivp_batch_parallel export
  odes/
    pendulum_ode.pyx  ODE in Cython syntax (noexcept nogil)
    setup.py          Cython build config
```

**Build and run:**

```bash
cd experiments/zig/zig_cython
zig build -Doptimize=ReleaseFast
cd odes && uv run cythonize -i pendulum_ode.pyx && cd ..
uv run python benchmark.py
```

### zig_mypyc/

mypyc (bundled with mypy) compiles typed Python to CPython C extensions. Unlike Cython,
mypyc cannot export a raw C function pointer with a custom calling convention -- it always
produces CPython extension modules that go through the Python object layer. So the ODE is
still called via a ctypes callback trampoline, same as the pure Python variant.

The result (~108s) is essentially identical to the pure Python callback (~105s). This
confirms that the bottleneck is the ctypes trampoline overhead per call, not the ODE
math itself. Compiling the ODE body to C with mypyc doesn't help when the FFI boundary
crossing dominates.

```
zig_mypyc/
  build.zig
  benchmark.py        Benchmark script (runs 100 ICs, extrapolates to 10k)
  src/
    solver_lib.zig    Same Dopri5 solver (shared library)
  odes/
    pendulum_ode.py   Typed Python ODE function
    setup.py          mypyc build config
```

**Build and run:**

```bash
cd experiments/zig/zig_mypyc
zig build -Doptimize=ReleaseFast
cd odes && uv run mypyc pendulum_ode.py && cd ..
uv run python benchmark.py
```

## Why Zig + Cython is ~20% slower than Zig native

Two factors account for the overhead:

1. **No inlining.** The native solver calls the ODE through a comptime-generic function,
   so the compiler inlines the ODE body directly into the solver loop. The Cython variant
   calls through an indirect C function pointer (`COdeFn`), which prevents inlining and
   adds branch-predictor overhead on every RK stage evaluation.

2. **Per-IC arena overhead.** The library solver (`solveSingle`) creates and destroys a
   fresh `ArenaAllocator` for each IC. The native solver reuses a single arena per thread
   with `arena.reset(.retain_capacity)`, avoiding repeated mmap/munmap syscalls.

## Why mypyc doesn't help

Both the mypyc and pure Python variants take ~105-108s. The ODE function is trivially
small (a few multiplications and a `sin()` call), so even in CPython it runs in
nanoseconds. The dominant cost is the ctypes callback trampoline: each of the ~7 RK
stage evaluations per step crosses the C-to-Python FFI boundary, acquires thread state,
unpacks pointer arguments, calls the Python function, and returns. Compiling the function
body to C with mypyc eliminates only the bytecode interpretation, which is a negligible
fraction of the per-call cost.

Cython avoids this entirely by exporting a raw `nogil` C function pointer that Zig calls
directly -- no trampoline, no Python runtime involvement, no GIL.
