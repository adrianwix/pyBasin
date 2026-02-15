const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

/// Function signature for an ODE right-hand side: f(t, y, dydt, params)
///
/// In Zig, we use an "out parameter" pattern instead of returning arrays:
/// - `t`: current time
/// - `y`: current state vector (input, not modified)
/// - `dydt`: output buffer where we write the derivatives
/// - `params`: problem-specific parameters (e.g., damping, forcing)
///
/// This avoids heap allocations in the hot loop — the caller allocates
/// buffers once and reuses them for millions of evaluations.
pub fn OdeFn(comptime ParamsT: type) type {
    return *const fn (t: f64, y: []const f64, dydt: []f64, params: *const ParamsT) void;
}

/// Dormand-Prince 5(4) adaptive step-size Runge-Kutta integrator.
///
/// How Dopri5 works:
/// - Each step evaluates the ODE right-hand side at 7 carefully chosen points
/// - Combines these to get a 5th-order accurate estimate of y(t+dt)
/// - Also computes a 4th-order estimate to measure local error
/// - If error is too large, rejects the step and tries smaller dt
/// - If error is small, accepts and increases dt for efficiency
///
/// This is a generic type in Zig — `comptime ParamsT` means the solver
/// is specialized at compile-time for your specific parameter type.
pub fn Dopri5(comptime ParamsT: type) type {
    return struct {
        const Self = @This();

        // Butcher tableau coefficients
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

        // Time fractions
        const c2: f64 = 1.0 / 5.0;
        const c3: f64 = 3.0 / 10.0;
        const c4: f64 = 4.0 / 5.0;
        const c5: f64 = 8.0 / 9.0;

        // Error coefficients (5th order - 4th order)
        // y_err = dt * (e1*k1 + e3*k3 + ... + e7*k7)
        // This is the difference between the 5th and 4th order solutions,
        // which tells us the local truncation error.
        const e1: f64 = 71.0 / 57600.0;
        const e3: f64 = -71.0 / 16695.0;
        const e4: f64 = 71.0 / 1920.0;
        const e5: f64 = -17253.0 / 339200.0;
        const e6: f64 = 22.0 / 525.0;
        const e7: f64 = -1.0 / 40.0;

        // Solver configuration parameters
        rtol: f64, // Relative tolerance (e.g., 1e-8)
        atol: f64, // Absolute tolerance (e.g., 1e-6)
        max_steps: usize, // Safety limit to prevent infinite loops

        // Step size adaptation parameters
        safety: f64 = 0.9, // Safety factor < 1 to be conservative
        min_factor: f64 = 0.2, // Don't shrink dt by more than 5x per step
        max_factor: f64 = 10.0, // Don't grow dt by more than 10x per step

        pub fn init(rtol: f64, atol: f64, max_steps: usize) Self {
            return .{
                .rtol = rtol,
                .atol = atol,
                .max_steps = max_steps,
            };
        }

        /// Integrate the ODE from t0 to t1, saving the solution at the specified
        /// evaluation times (save_at). Uses dense output (cubic Hermite interpolation)
        /// to evaluate at arbitrary times without reducing the step size.
        ///
        /// Returns a 2D array of shape [save_at.len][dim] with the solution at each
        /// requested time point.
        pub fn solve(
            self: *const Self,
            ode_fn: OdeFn(ParamsT),
            params: *const ParamsT,
            y0: []const f64,
            t0: f64,
            t1: f64,
            save_at: []const f64,
            allocator: Allocator,
        ) ![][]f64 {
            const dim = y0.len;

            // ========== Memory Management (Zig pattern) ==========
            // In Zig, we manually manage memory with allocators.
            // The pattern: allocate, then immediately `defer free` so cleanup
            // happens automatically when this function returns.

            // Working buffers for state vectors
            const y = try allocator.alloc(f64, dim); // Current state
            defer allocator.free(y);
            const y_new = try allocator.alloc(f64, dim); // Trial state (5th order)
            defer allocator.free(y_new);
            const y_err = try allocator.alloc(f64, dim); // Error estimate
            defer allocator.free(y_err);

            // ========== The FSAL Trick ==========
            // FSAL = "First Same As Last"
            // Dopri5 has a special property: the last evaluation k7 of step n
            // can be reused as k1 for step n+1 (they're at the same point).
            // This saves 1 function evaluation per step (6 instead of 7).
            //
            // We allocate 7 buffers but only use 6 per step by swapping indices:
            // - k_bufs[0] and k_bufs[6] swap roles as k1 and k7
            // - k_bufs[1..5] are always k2..k6
            var k_bufs: [7][]f64 = undefined;
            for (&k_bufs) |*kb| {
                kb.* = try allocator.alloc(f64, dim);
            }
            defer for (&k_bufs) |kb| {
                allocator.free(kb);
            };
            var k1_idx: usize = 0; // Index of k1 buffer (starts at 0)
            var k7_idx: usize = 6; // Index of k7 buffer (starts at 6)

            const y_stage = try allocator.alloc(f64, dim); // Temp for intermediate stages
            defer allocator.free(y_stage);

            // Allocate output array
            // result[i] will hold the solution at time save_at[i]
            const result = try allocator.alloc([]f64, save_at.len);
            for (result) |*row| {
                row.* = try allocator.alloc(f64, dim);
            }

            // ========== Initialize Integration ==========
            @memcpy(y, y0); // Copy initial conditions into working buffer

            var t = t0;
            var dt = self.initialStepSize(t0, t1); // Start with a small conservative step
            var save_idx: usize = 0; // Index into save_at[] array

            // Evaluate k1 = f(t0, y0) once before the main loop starts
            ode_fn(t, y, k_bufs[k1_idx], params);

            // ========== Main Integration Loop ==========
            var step_count: usize = 0;
            while (t < t1 and step_count < self.max_steps) : (step_count += 1) {
                // Don't overshoot the final time
                if (t + dt > t1) dt = t1 - t;
                if (dt <= 0) break;

                // Get pointers to the k buffers (cleaner syntax for the stages below)
                const k1 = k_bufs[k1_idx];
                const k2 = k_bufs[1];
                const k3 = k_bufs[2];
                const k4 = k_bufs[3];
                const k5 = k_bufs[4];
                const k6 = k_bufs[5];
                const k7 = k_bufs[k7_idx];

                // ===== Dopri5 Stages =====
                // Each stage evaluates the ODE at a different point within [t, t+dt]
                // using a weighted combination of previous slopes (k1, k2, ...)

                // Stage 2: evaluate at t + dt/5
                for (0..dim) |i| {
                    y_stage[i] = y[i] + dt * a21 * k1[i];
                }
                ode_fn(t + c2 * dt, y_stage, k2, params);

                // Stage 3: evaluate at t + 3*dt/10
                for (0..dim) |i| {
                    y_stage[i] = y[i] + dt * (a31 * k1[i] + a32 * k2[i]);
                }
                ode_fn(t + c3 * dt, y_stage, k3, params);

                // Stage 4: evaluate at t + 4*dt/5
                for (0..dim) |i| {
                    y_stage[i] = y[i] + dt * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
                }
                ode_fn(t + c4 * dt, y_stage, k4, params);

                // Stage 5: evaluate at t + 8*dt/9
                for (0..dim) |i| {
                    y_stage[i] = y[i] + dt * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
                }
                ode_fn(t + c5 * dt, y_stage, k5, params);

                // Stage 6: evaluate at t + dt
                for (0..dim) |i| {
                    y_stage[i] = y[i] + dt * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
                }
                ode_fn(t + dt, y_stage, k6, params);

                // Compute the 5th-order solution
                // This is our best estimate of y(t+dt)
                for (0..dim) |i| {
                    y_new[i] = y[i] + dt * (a71 * k1[i] + a73 * k3[i] + a74 * k4[i] + a75 * k5[i] + a76 * k6[i]);
                }

                // Stage 7: evaluate at the new point
                // This both gives us k7 for error estimation AND becomes k1 for
                // the next step (FSAL trick)
                ode_fn(t + dt, y_new, k7, params);

                // ===== Error Estimation =====
                // Compute y_err = (5th order solution) - (4th order solution)
                // The embedded 4th-order formula uses different weights
                for (0..dim) |i| {
                    y_err[i] = dt * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
                }

                // ===== Compute Error Norm =====
                // We need a single number to measure how "bad" the error is.
                // Scale each component by atol + rtol*|y| to handle both
                // small (absolute) and large (relative) values properly.
                var err_norm: f64 = 0.0;
                for (0..dim) |i| {
                    const sc = self.atol + self.rtol * @max(@abs(y[i]), @abs(y_new[i]));
                    const ratio = y_err[i] / sc;
                    err_norm += ratio * ratio; // RMS error
                }
                err_norm = @sqrt(err_norm / @as(f64, @floatFromInt(dim)));

                // ===== Step Acceptance / Rejection =====
                // If err_norm <= 1, the error is within tolerance -> accept
                // If err_norm > 1, the error is too large -> reject and retry with smaller dt
                if (err_norm <= 1.0) {
                    // STEP ACCEPTED!

                    // Save solution at any requested output times in [t, t+dt]
                    const t_new = t + dt;
                    while (save_idx < save_at.len and save_at[save_idx] <= t_new + 1e-12) {
                        if (save_at[save_idx] <= t + 1e-12) {
                            // Output time is at the start of the step
                            @memcpy(result[save_idx], y);
                        } else {
                            // Output time is inside [t, t+dt] — use dense output
                            // (interpolation using the k values we already computed)
                            const th = (save_at[save_idx] - t) / dt; // theta in [0,1]
                            self.denseOutput(th, dt, y, k1, k3, k4, k5, k6, k7, result[save_idx], dim);
                        }
                        save_idx += 1;
                    }

                    // Move forward: y becomes y_new, t becomes t_new
                    @memcpy(y, y_new);
                    t = t_new;

                    // FSAL trick: swap indices so k7 becomes k1 for next iteration
                    const tmp = k1_idx;
                    k1_idx = k7_idx;
                    k7_idx = tmp;
                }

                // ===== Adaptive Step Size Control =====
                // Increase dt if error is small, decrease if large.
                // Factor is chosen based on err_norm^(-1/5) (optimal for 5th-order method)
                const factor = if (err_norm == 0.0)
                    self.max_factor // Perfect accuracy -> grow maximally
                else
                    // safety * err_norm^(-0.2) with bounds [min_factor, max_factor]
                    @min(self.max_factor, @max(self.min_factor, self.safety * math.pow(f64, err_norm, -0.2)));
                dt *= factor;

                // Note: If step was rejected (err_norm > 1), we stay at the same t
                // but retry with smaller dt. If accepted, we advanced t already.
            }

            // Fill any remaining save_at points with the final state
            // (handles the case where save_at[i] > t1)
            while (save_idx < save_at.len) : (save_idx += 1) {
                @memcpy(result[save_idx], y);
            }

            return result;
        }

        /// Choose initial step size (conservative estimate).
        fn initialStepSize(self: *const Self, t0: f64, t1: f64) f64 {
            _ = self;
            return (t1 - t0) * 1e-3; // Start with 0.1% of the integration interval
        }

        /// Dense output: evaluate the solution at any time within a completed step.
        ///
        /// We've already computed k1..k7 for the interval [t, t+dt].
        /// This uses a 4th-order polynomial interpolant (Shampine's formula)
        /// to get y(t + th*dt) for any th in [0, 1] WITHOUT re-evaluating the ODE.
        ///
        /// Why this matters: if you want output at 100 evenly spaced times,
        /// dense output lets the solver take large adaptive steps and interpolate
        /// afterward, rather than forcing it to hit every output time exactly
        /// (which would ruin the adaptive step size benefits).
        fn denseOutput(
            _: *const Self,
            th: f64, // theta in [0,1]: fraction of the step
            dt: f64, // step size
            y: []const f64,
            k1: []const f64,
            k3: []const f64,
            k4: []const f64,
            k5: []const f64,
            k6: []const f64,
            k7: []const f64,
            out: []f64, // output buffer
            dim: usize,
        ) void {
            // Polynomial coefficients (derived by Shampine for Dopri5)
            const b1 = th * (1.0 + th * (-1337.0 / 480.0 + th * (1039.0 / 360.0 - th * 1163.0 / 1152.0)));
            const b3 = th * th * (100.0 / 63.0 + th * (-536.0 / 189.0 + th * 2507.0 / 2016.0));
            const b4 = th * th * (-125.0 / 96.0 + th * (2875.0 / 1152.0 - th * 13411.0 / 12288.0));
            const b5 = th * th * (3567.0 / 14336.0 + th * (-24111.0 / 57344.0 + th * 16737.0 / 90112.0));
            const b6 = th * th * (-11.0 / 70.0 + th * (187.0 / 630.0 - th * 11.0 / 84.0));
            const b7 = th * th * th * (-11.0 / 40.0 + th * 11.0 / 40.0);

            // Compute y(t + th*dt) = y(t) + dt * (weighted sum of slopes)
            for (0..dim) |i| {
                out[i] = y[i] + dt * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i] + b7 * k7[i]);
            }
        }

        /// Free the result array returned by solve().
        ///
        /// In Zig, memory management is explicit: whoever allocates must free.
        /// The `solve()` function returns a 2D array ([][]f64) that was allocated
        /// in multiple pieces: one outer array and multiple inner arrays.
        /// This helper frees all of them in the correct order.
        pub fn freeResult(result: [][]f64, allocator: Allocator) void {
            for (result) |row| {
                allocator.free(row); // Free each inner array
            }
            allocator.free(result); // Free the outer array
        }
    };
}
