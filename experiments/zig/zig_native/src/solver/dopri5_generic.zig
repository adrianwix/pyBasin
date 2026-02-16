const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

/// Generic ODE function type for C ABI compatibility.
/// This allows loading ODE functions from external shared libraries.
/// Uses opaque params pointer instead of compile-time generic.
pub const GenericOdeFn = *const fn (t: f64, y: [*]const f64, dydt: [*]f64, params: ?*const anyopaque, dim: usize) callconv(.c) void;

/// Dormand-Prince 5(4) adaptive step-size Runge-Kutta integrator.
/// This version uses runtime dispatch for ODEs (opaque params pointer).
pub const Dopri5Generic = struct {
    const Self = @This();

    // Butcher tableau coefficients (same as typed version)
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

    rtol: f64,
    atol: f64,
    max_steps: usize,
    safety: f64 = 0.9,
    min_factor: f64 = 0.2,
    max_factor: f64 = 10.0,

    pub fn init(rtol: f64, atol: f64, max_steps: usize) Self {
        return .{
            .rtol = rtol,
            .atol = atol,
            .max_steps = max_steps,
        };
    }

    pub fn solve(
        self: *const Self,
        ode_fn: GenericOdeFn,
        params: ?*const anyopaque,
        y0: []const f64,
        dim: usize,
        t0: f64,
        t1: f64,
        save_at: []const f64,
        allocator: Allocator,
    ) ![][]f64 {
        // Working buffers
        const y = try allocator.alloc(f64, dim);
        defer allocator.free(y);
        const y_new = try allocator.alloc(f64, dim);
        defer allocator.free(y_new);
        const y_err = try allocator.alloc(f64, dim);
        defer allocator.free(y_err);

        // FSAL buffers
        var k_bufs: [7][]f64 = undefined;
        for (&k_bufs) |*kb| {
            kb.* = try allocator.alloc(f64, dim);
        }
        defer for (&k_bufs) |kb| {
            allocator.free(kb);
        };
        var k1_idx: usize = 0;
        var k7_idx: usize = 6;

        const y_stage = try allocator.alloc(f64, dim);
        defer allocator.free(y_stage);

        // Output array
        const result = try allocator.alloc([]f64, save_at.len);
        for (result) |*row| {
            row.* = try allocator.alloc(f64, dim);
        }

        // Initialize
        @memcpy(y, y0);
        var t = t0;
        var dt = self.initialStepSize(t0, t1);
        var save_idx: usize = 0;

        // Initial k1
        ode_fn(t, y.ptr, k_bufs[k1_idx].ptr, params, dim);

        // Main loop
        var step_count: usize = 0;
        while (t < t1 and step_count < self.max_steps) : (step_count += 1) {
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

            // Stage 7
            ode_fn(t + dt, y_new.ptr, k7.ptr, params, dim);

            // Error estimate
            for (0..dim) |i| {
                y_err[i] = dt * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
            }

            // Error norm
            var err_norm: f64 = 0.0;
            for (0..dim) |i| {
                const sc = self.atol + self.rtol * @max(@abs(y[i]), @abs(y_new[i]));
                const ratio = y_err[i] / sc;
                err_norm += ratio * ratio;
            }
            err_norm = @sqrt(err_norm / @as(f64, @floatFromInt(dim)));

            if (err_norm <= 1.0) {
                const t_new = t + dt;
                while (save_idx < save_at.len and save_at[save_idx] <= t_new + 1e-12) {
                    if (save_at[save_idx] <= t + 1e-12) {
                        @memcpy(result[save_idx], y);
                    } else {
                        const th = (save_at[save_idx] - t) / dt;
                        self.denseOutput(th, dt, y, k1, k3, k4, k5, k6, k7, result[save_idx], dim);
                    }
                    save_idx += 1;
                }

                @memcpy(y, y_new);
                t = t_new;

                const tmp = k1_idx;
                k1_idx = k7_idx;
                k7_idx = tmp;
            }

            const factor = if (err_norm == 0.0)
                self.max_factor
            else
                @min(self.max_factor, @max(self.min_factor, self.safety * math.pow(f64, err_norm, -0.2)));
            dt *= factor;
        }

        while (save_idx < save_at.len) : (save_idx += 1) {
            @memcpy(result[save_idx], y);
        }

        return result;
    }

    fn initialStepSize(self: *const Self, t0: f64, t1: f64) f64 {
        _ = self;
        return (t1 - t0) * 1e-3;
    }

    fn denseOutput(
        _: *const Self,
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

    pub fn freeResult(result: [][]f64, allocator: Allocator) void {
        for (result) |row| {
            allocator.free(row);
        }
        allocator.free(result);
    }
};
