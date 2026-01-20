# Solver Comparison

!!! note "Documentation in Progress"
This page is under construction.

## Test Configuration

- ODE: Driven damped pendulum
- t_span: (0, 1000)
- n_steps: 1000
- Tolerances: rtol=1e-8, atol=1e-6

## Results

| Solver             | Device | Time (s) | ms/integration | Speedup vs MATLAB | BS Valid |
| ------------------ | ------ | -------- | -------------- | ----------------- | -------- |
| torchdiffeq_rk4    | cpu    | 0.156    | 0.016          | 779x              | ❌       |
| torchdiffeq_dopri5 | cpu    | 8.10     | 0.81           | 15x               | ✅       |
| jax_diffrax_dopri5 | cpu    | 9.33     | 0.93           | 13x               | ✅       |
| matlab_ode45       | cpu    | 11.33    | 1.13           | baseline          | ✅       |
| jax_diffrax_dopri5 | cuda   | 11.57    | 1.16           | 10.5x             | ✅       |
| jax_diffrax_tsit5  | cpu    | 34.03    | 3.40           | 3.6x              | ✅       |

## Key Findings

- RK4 is fast but produces incorrect BS (insufficient accuracy)
- Dopri5 methods are ~10-15x faster than MATLAB ode45
- GPU benefit appears at larger sample sizes
