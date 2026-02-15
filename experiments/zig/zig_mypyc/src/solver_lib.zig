const std = @import("std");
const math = std.math;

/// C-compatible ODE function pointer.
///
/// This is the callback signature that Python must match via ctypes.CFUNCTYPE.
/// When Zig calls this, execution crosses the FFI boundary into Python,
/// acquires the GIL, runs the Python function, and returns.
///
/// - `t`: current time
/// - `y`: pointer to state vector (length `dim`)
/// - `dydt`: output pointer for derivatives (length `dim`)
/// - `params`: pointer to parameter array (length is ODE-specific)
/// - `dim`: number of state dimensions
const COdeFn = *const fn (
    t: f64,
    y: [*]const f64,
    dydt: [*]f64,
    params: [*]const f64,
    dim: usize,
) callconv(.c) void;

// ============================================================
// Butcher tableau for Dormand-Prince 5(4)
// ============================================================

const a21: f64 = 1.0 / 5.0;
const a31: f64 = 3.0 / 40.0;
const a32: f64 = 9.0 / 40.0;
const a41: f64 = 44.0 / 45.0;
const a42: f64 = -56.0 / 15.0;
const a43: f64 = 32.0 / 9.0;
const a51: f64 = 19372.0 / 6561.0;
const a52: f64 = -25360.0 / 2187.0;
const a53: f64 = 64448.0 / 6561.0;
const a54: f64 = -212.0 / 729.0;
const a61: f64 = 9017.0 / 3168.0;
const a62: f64 = -355.0 / 33.0;
const a63: f64 = 46732.0 / 5247.0;
const a64: f64 = 49.0 / 176.0;
const a65: f64 = -5103.0 / 18656.0;
const a71: f64 = 35.0 / 384.0;
const a73: f64 = 500.0 / 1113.0;
const a74: f64 = 125.0 / 192.0;
const a75: f64 = -2187.0 / 6784.0;
const a76: f64 = 11.0 / 84.0;

const c2: f64 = 1.0 / 5.0;
const c3: f64 = 3.0 / 10.0;
const c4: f64 = 4.0 / 5.0;
const c5: f64 = 8.0 / 9.0;

const e1: f64 = 71.0 / 57600.0;
const e3: f64 = -71.0 / 16695.0;
const e4: f64 = 71.0 / 1920.0;
const e5: f64 = -17253.0 / 339200.0;
const e6: f64 = 22.0 / 525.0;
const e7: f64 = -1.0 / 40.0;

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;

/// Dense output interpolation (Shampine's formula for Dopri5).
///
/// Evaluates the solution at any point within a completed step
/// using the already-computed k1..k7 values — no extra ODE evaluations needed.
fn denseOutput(
    th: f64,
    dt: f64,
    y: []const f64,
    k1: []const f64,
    k3: []const f64,
    k4: []const f64,
    k5: []const f64,
    k6: []const f64,
    k7: []const f64,
    out: []f64,
    dim: usize,
) void {
    const b1 = th * (1.0 + th * (-1337.0 / 480.0 + th * (1039.0 / 360.0 - th * 1163.0 / 1152.0)));
    const b3 = th * th * (100.0 / 63.0 + th * (-536.0 / 189.0 + th * 2507.0 / 2016.0));
    const b4 = th * th * (-125.0 / 96.0 + th * (2875.0 / 1152.0 - th * 13411.0 / 12288.0));
    const b5 = th * th * (3567.0 / 14336.0 + th * (-24111.0 / 57344.0 + th * 16737.0 / 90112.0));
    const b6 = th * th * (-11.0 / 70.0 + th * (187.0 / 630.0 - th * 11.0 / 84.0));
    const b7 = th * th * th * (-11.0 / 40.0 + th * 11.0 / 40.0);

    for (0..dim) |i| {
        out[i] = y[i] + dt * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i] + b7 * k7[i]);
    }
}

/// Solve a single IVP using Dopri5 with a C-callable ODE callback.
///
/// Results are written to `result_ptr` in row-major order:
///   result_ptr[i * dim + j] = component j at time t_eval[i]
///
/// Returns 0 on success, -1 on allocation failure.
fn solveSingle(
    ode_fn: COdeFn,
    params: [*]const f64,
    y0: [*]const f64,
    dim: usize,
    t0: f64,
    t1: f64,
    rtol: f64,
    atol: f64,
    t_eval: [*]const f64,
    n_eval: usize,
    result_ptr: [*]f64,
    max_steps: usize,
) i32 {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // Allocate working buffers
    const y = alloc.alloc(f64, dim) catch return -1;
    const y_new = alloc.alloc(f64, dim) catch return -1;
    const y_err = alloc.alloc(f64, dim) catch return -1;
    const y_stage = alloc.alloc(f64, dim) catch return -1;

    // 7 k-buffers for FSAL Dopri5
    var k_bufs: [7][]f64 = undefined;
    for (&k_bufs) |*kb| {
        kb.* = alloc.alloc(f64, dim) catch return -1;
    }

    var k1_idx: usize = 0;
    var k7_idx: usize = 6;

    // Copy initial conditions
    @memcpy(y, y0[0..dim]);

    var t = t0;
    var dt = (t1 - t0) * 1e-3;
    var save_idx: usize = 0;

    // First evaluation: k1 = f(t0, y0)
    ode_fn(t, y.ptr, k_bufs[k1_idx].ptr, params, dim);

    var step_count: usize = 0;
    while (t < t1 and step_count < max_steps) : (step_count += 1) {
        if (t + dt > t1) dt = t1 - t;
        if (dt <= 0) break;

        const k1 = k_bufs[k1_idx];
        const k2 = k_bufs[1];
        const k3 = k_bufs[2];
        const k4 = k_bufs[3];
        const k5 = k_bufs[4];
        const k6 = k_bufs[5];
        const k7 = k_bufs[k7_idx];

        // Stage 2
        for (0..dim) |i| {
            y_stage[i] = y[i] + dt * a21 * k1[i];
        }
        ode_fn(t + c2 * dt, y_stage.ptr, k2.ptr, params, dim);

        // Stage 3
        for (0..dim) |i| {
            y_stage[i] = y[i] + dt * (a31 * k1[i] + a32 * k2[i]);
        }
        ode_fn(t + c3 * dt, y_stage.ptr, k3.ptr, params, dim);

        // Stage 4
        for (0..dim) |i| {
            y_stage[i] = y[i] + dt * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
        }
        ode_fn(t + c4 * dt, y_stage.ptr, k4.ptr, params, dim);

        // Stage 5
        for (0..dim) |i| {
            y_stage[i] = y[i] + dt * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
        }
        ode_fn(t + c5 * dt, y_stage.ptr, k5.ptr, params, dim);

        // Stage 6
        for (0..dim) |i| {
            y_stage[i] = y[i] + dt * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
        }
        ode_fn(t + dt, y_stage.ptr, k6.ptr, params, dim);

        // 5th-order solution
        for (0..dim) |i| {
            y_new[i] = y[i] + dt * (a71 * k1[i] + a73 * k3[i] + a74 * k4[i] + a75 * k5[i] + a76 * k6[i]);
        }

        // Stage 7 (FSAL — becomes k1 for next step)
        ode_fn(t + dt, y_new.ptr, k7.ptr, params, dim);

        // Error estimate
        for (0..dim) |i| {
            y_err[i] = dt * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
        }

        // Error norm (RMS)
        var err_norm: f64 = 0.0;
        for (0..dim) |i| {
            const sc = atol + rtol * @max(@abs(y[i]), @abs(y_new[i]));
            const ratio = y_err[i] / sc;
            err_norm += ratio * ratio;
        }
        err_norm = @sqrt(err_norm / @as(f64, @floatFromInt(dim)));

        if (err_norm <= 1.0) {
            const t_new = t + dt;

            // Save at requested output times via dense interpolation
            while (save_idx < n_eval and t_eval[save_idx] <= t_new + 1e-12) {
                const out_slice = result_ptr[save_idx * dim .. save_idx * dim + dim];
                if (t_eval[save_idx] <= t + 1e-12) {
                    @memcpy(out_slice, y);
                } else {
                    const th = (t_eval[save_idx] - t) / dt;
                    denseOutput(th, dt, y, k1, k3, k4, k5, k6, k7, out_slice, dim);
                }
                save_idx += 1;
            }

            @memcpy(y, y_new);
            t = t_new;

            // FSAL swap
            const tmp = k1_idx;
            k1_idx = k7_idx;
            k7_idx = tmp;
        }

        // Adaptive step size
        const factor = if (err_norm == 0.0)
            MAX_FACTOR
        else
            @min(MAX_FACTOR, @max(MIN_FACTOR, SAFETY * math.pow(f64, err_norm, -0.2)));
        dt *= factor;
    }

    // Fill remaining save points with final state
    while (save_idx < n_eval) : (save_idx += 1) {
        const out_slice = result_ptr[save_idx * dim .. save_idx * dim + dim];
        @memcpy(out_slice, y);
    }

    return 0;
}

// ============================================================
// Exported C ABI functions
// ============================================================

/// Solve a single IVP using a raw function pointer (usize).
/// Python passes the Cython function address as integer — no ctypes trampoline.
export fn solve_ivp_raw(
    ode_fn_ptr: usize,
    params: [*]const f64,
    y0: [*]const f64,
    dim: usize,
    t0: f64,
    t1: f64,
    rtol: f64,
    atol: f64,
    t_eval: [*]const f64,
    n_eval: usize,
    result_ptr: [*]f64,
    max_steps: usize,
) callconv(.c) i32 {
    const ode_fn: COdeFn = @ptrFromInt(ode_fn_ptr);
    return solveSingle(ode_fn, params, y0, dim, t0, t1, rtol, atol, t_eval, n_eval, result_ptr, max_steps);
}

/// Solve a single initial value problem.
///
/// `ode_fn` is a C-compatible function pointer (e.g. from ctypes.CFUNCTYPE).
/// Result is written to `result_ptr` in row-major layout [n_eval × dim].
export fn solve_ivp(
    ode_fn: COdeFn,
    params: [*]const f64,
    y0: [*]const f64,
    dim: usize,
    t0: f64,
    t1: f64,
    rtol: f64,
    atol: f64,
    t_eval: [*]const f64,
    n_eval: usize,
    result_ptr: [*]f64,
    max_steps: usize,
) callconv(.c) i32 {
    return solveSingle(ode_fn, params, y0, dim, t0, t1, rtol, atol, t_eval, n_eval, result_ptr, max_steps);
}

/// Solve a batch of IVPs sequentially.
///
/// - `ics_ptr`: row-major [n_ics × dim] initial conditions
/// - `result_ptr`: row-major [n_ics × n_eval × dim] output
export fn solve_ivp_batch(
    ode_fn: COdeFn,
    params: [*]const f64,
    ics_ptr: [*]const f64,
    n_ics: usize,
    dim: usize,
    t0: f64,
    t1: f64,
    rtol: f64,
    atol: f64,
    t_eval: [*]const f64,
    n_eval: usize,
    result_ptr: [*]f64,
    max_steps: usize,
) callconv(.c) i32 {
    const stride = n_eval * dim;
    for (0..n_ics) |i| {
        const y0 = ics_ptr + i * dim;
        const out = result_ptr + i * stride;
        const ret = solveSingle(ode_fn, params, y0, dim, t0, t1, rtol, atol, t_eval, n_eval, out, max_steps);
        if (ret != 0) return ret;
    }
    return 0;
}

/// Multi-threaded batch solver for compiled (GIL-free) ODE callbacks.
///
/// `ode_fn_ptr` is a raw function pointer address (Python integer cast to usize).
/// This bypasses ctypes trampoline entirely — Zig casts the address directly
/// to a native function pointer with zero overhead per call.
///
/// Use this when the ODE is compiled via Cython/Numba to native C.
/// The `nogil` ODE is safe to call from any thread without Python state.
export fn solve_ivp_batch_parallel(
    ode_fn_ptr: usize,
    params: [*]const f64,
    ics_ptr: [*]const f64,
    n_ics: usize,
    dim: usize,
    t0: f64,
    t1: f64,
    rtol: f64,
    atol: f64,
    t_eval: [*]const f64,
    n_eval: usize,
    result_ptr: [*]f64,
    max_steps: usize,
    n_threads: usize,
) callconv(.c) i32 {
    // Cast the raw integer address to a proper C function pointer.
    // This is the key: we bypass ctypes entirely and call the Cython-compiled
    // C function directly from Zig threads.
    const ode_fn: COdeFn = @ptrFromInt(ode_fn_ptr);

    const actual_threads = if (n_threads == 0) (std.Thread.getCpuCount() catch 4) else n_threads;
    const chunk = (n_ics + actual_threads - 1) / actual_threads;
    const stride = n_eval * dim;

    var threads = std.heap.page_allocator.alloc(std.Thread, actual_threads) catch return -1;
    defer std.heap.page_allocator.free(threads);

    var error_flag: i32 = 0;
    var spawned: usize = 0;

    for (0..actual_threads) |t_idx| {
        const start = t_idx * chunk;
        const end = @min(start + chunk, n_ics);
        if (start >= end) break;

        threads[t_idx] = std.Thread.spawn(.{}, struct {
            fn worker(
                ofn: COdeFn,
                p: [*]const f64,
                ics: [*]const f64,
                d: usize,
                t0_: f64,
                t1_: f64,
                rtol_: f64,
                atol_: f64,
                te: [*]const f64,
                ne: usize,
                res: [*]f64,
                ms: usize,
                s: usize,
                e: usize,
                strd: usize,
                eflag: *i32,
            ) void {
                for (s..e) |i| {
                    const y0 = ics + i * d;
                    const out = res + i * strd;
                    const ret = solveSingle(ofn, p, y0, d, t0_, t1_, rtol_, atol_, te, ne, out, ms);
                    if (ret != 0) {
                        @atomicStore(i32, eflag, ret, .release);
                    }
                }
            }
        }.worker, .{
            ode_fn, params, ics_ptr,    dim,
            t0,     t1,     rtol,       atol,
            t_eval, n_eval, result_ptr, max_steps,
            start,  end,    stride,     &error_flag,
        }) catch return -1;
        spawned += 1;
    }

    for (threads[0..spawned]) |th| {
        th.join();
    }

    return @atomicLoad(i32, &error_flag, .acquire);
}
