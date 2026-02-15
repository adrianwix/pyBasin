const std = @import("std");
const math = std.math;

/// Parameters for the pendulum ODE system.
///
/// This is a Zig struct — similar to a Python dataclass or TypedDict.
/// Fields can have default values (like Python default arguments).
pub const PendulumParams = struct {
    alpha: f64 = 0.1, // Damping coefficient
    T: f64 = 0.5, // External torque
    K: f64 = 1.0, // Stiffness coefficient
};

/// Pendulum ODE right-hand side: dθ/dt = θ̇, dθ̇/dt = -α·θ̇ + T - K·sin(θ)
///
/// This implements the equations of motion for a damped, driven pendulum.
///
/// Zig pattern: we use an "out-parameter" (dydt) instead of returning a value.
/// This avoids allocating memory on every call — the caller allocates once
/// and reuses the buffer millions of times during integration.
///
/// Arguments:
///   t:      Current time (not used in this autonomous system)
///   y:      State vector [θ, θ̇] (input, not modified)
///   dydt:   Output buffer where we write [dθ/dt, dθ̇/dt]
///   params: Parameters (damping, torque, stiffness)
pub fn pendulum_ode(t: f64, y: []const f64, dydt: []f64, params: *const PendulumParams) void {
    _ = t; // Zig requires acknowledging unused parameters with `_ = ...`

    const theta = y[0]; // Angle
    const theta_dot = y[1]; // Angular velocity

    // dθ/dt = θ̇
    dydt[0] = theta_dot;

    // dθ̇/dt = -α·θ̇ + T - K·sin(θ)
    // This is Newton's 2nd law for rotational motion with:
    // - Damping: -α·θ̇ (friction)
    // - Forcing: T (constant external torque)
    // - Restoring: -K·sin(θ) (gravity/spring-like force)
    dydt[1] = -params.alpha * theta_dot + params.T - params.K * @sin(theta);
}
