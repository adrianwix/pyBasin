const std = @import("std");
const math = std.math;
const Thread = std.Thread;
const Allocator = std.mem.Allocator;
const dopri5_mod = @import("dopri5.zig");
const pendulum = @import("pendulum.zig");

const Dopri5 = dopri5_mod.Dopri5(pendulum.PendulumParams);

const N_SAMPLES: usize = 10000;
const DIM: usize = 2;
const T0: f64 = 0.0;
const T1: f64 = 1000.0;
const T_STEADY: f64 = 900.0;
const N_SAVE_TOTAL: usize = 10000;
const FP_THRESHOLD: f64 = 0.01;

fn buildSaveAt() [N_SAVE_TOTAL]f64 {
    var pts: [N_SAVE_TOTAL]f64 = undefined;
    const dt = (T1 - T0) / @as(f64, @floatFromInt(N_SAVE_TOTAL - 1));
    for (0..N_SAVE_TOTAL) |i| {
        pts[i] = T0 + @as(f64, @floatFromInt(i)) * dt;
    }
    return pts;
}

fn generateInitialConditions(
    ics: [][]f64,
    params: *const pendulum.PendulumParams,
    seed: u64,
) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();
    const offset = math.asin(params.T / params.K);
    const theta_min = -math.pi + offset;
    const theta_max = math.pi + offset;

    for (ics) |ic| {
        ic[0] = theta_min + random.float(f64) * (theta_max - theta_min);
        ic[1] = -10.0 + random.float(f64) * 20.0;
    }
}

const Label = enum { FP, LC };

/// Classify a trajectory from its steady-state angular velocity time series.
/// delta = |max(theta_dot) - mean(theta_dot)| < threshold => FP, else LC
/// Only looks at the steady-state portion (t >= T_STEADY) of the trajectory.
fn classify(sol: [][]f64) Label {
    // Calculate which index corresponds to t >= T_STEADY
    // We have N_SAVE_TOTAL points from T0 to T1
    const steady_idx = @as(usize, @intFromFloat(@ceil((T_STEADY - T0) / (T1 - T0) * @as(f64, @floatFromInt(N_SAVE_TOTAL - 1)))));

    // Only analyze the steady-state portion
    const steady_sol = sol[steady_idx..];

    var max_val: f64 = -math.inf(f64);
    var sum: f64 = 0.0;
    for (steady_sol) |state| {
        const theta_dot = state[1];
        if (theta_dot > max_val) max_val = theta_dot;
        sum += theta_dot;
    }
    const mean_val = sum / @as(f64, @floatFromInt(steady_sol.len));
    const delta = @abs(max_val - mean_val);
    return if (delta < FP_THRESHOLD) .FP else .LC;
}

const WorkerCtx = struct {
    ics: [][]f64,
    save_at: []const f64,
    labels: []Label,
    params: *const pendulum.PendulumParams,
    solver: *const Dopri5,
    start: usize,
    end: usize,
};

fn workerFn(ctx: WorkerCtx) void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    for (ctx.start..ctx.end) |i| {
        const sol = ctx.solver.solve(
            pendulum.pendulum_ode,
            ctx.params,
            ctx.ics[i],
            T0,
            T1,
            ctx.save_at,
            alloc,
        ) catch {
            ctx.labels[i] = .FP;
            continue;
        };
        ctx.labels[i] = classify(sol);
        _ = arena.reset(.retain_capacity);
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const params = pendulum.PendulumParams{
        .alpha = 0.1,
        .T = 0.5,
        .K = 1.0,
    };

    const solver = Dopri5.init(1e-8, 1e-6, 1_000_000);
    const save_at = buildSaveAt();

    const ics = try allocator.alloc([]f64, N_SAMPLES);
    defer {
        for (ics) |ic| allocator.free(ic);
        allocator.free(ics);
    }
    for (ics) |*ic| {
        ic.* = try allocator.alloc(f64, DIM);
    }

    const labels = try allocator.alloc(Label, N_SAMPLES);
    defer allocator.free(labels);

    generateInitialConditions(ics, &params, 42);

    const n_threads = Thread.getCpuCount() catch 4;
    const chunk = (N_SAMPLES + n_threads - 1) / n_threads;

    std.debug.print("Solving {d} pendulum ICs with Dopri5 on {d} threads...\n", .{ N_SAMPLES, n_threads });
    std.debug.print("Saving {d} points from t={d:.0} to t={d:.0} (using t>={d:.0} for classification)\n", .{ N_SAVE_TOTAL, T0, T1, T_STEADY });

    var timer = try std.time.Timer.start();

    var threads = try allocator.alloc(Thread, n_threads);
    defer allocator.free(threads);

    for (0..n_threads) |t| {
        const start = t * chunk;
        const end = @min(start + chunk, N_SAMPLES);
        if (start >= end) {
            threads[t] = try Thread.spawn(.{}, struct {
                fn noop() void {}
            }.noop, .{});
            continue;
        }
        threads[t] = try Thread.spawn(.{}, workerFn, .{WorkerCtx{
            .ics = ics,
            .save_at = &save_at,
            .labels = labels,
            .params = &params,
            .solver = &solver,
            .start = start,
            .end = end,
        }});
    }

    for (threads) |t| {
        t.join();
    }

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

    var fp_count: usize = 0;
    var lc_count: usize = 0;
    for (labels) |label| {
        switch (label) {
            .FP => fp_count += 1,
            .LC => lc_count += 1,
        }
    }

    const n_f: f64 = @floatFromInt(N_SAMPLES);
    const fp_frac = @as(f64, @floatFromInt(fp_count)) / n_f;
    const lc_frac = @as(f64, @floatFromInt(lc_count)) / n_f;

    std.debug.print("\nDone in {d:.1} ms ({d:.1} us per IC)\n", .{
        elapsed_ms,
        elapsed_ms * 1000.0 / n_f,
    });

    std.debug.print("\n=== Basin Stability ===\n", .{});
    std.debug.print("  Fixed Point (FP): {d:5} / {d}  =  {d:.4}\n", .{ fp_count, N_SAMPLES, fp_frac });
    std.debug.print("  Limit Cycle (LC): {d:5} / {d}  =  {d:.4}\n", .{ lc_count, N_SAMPLES, lc_frac });
}
