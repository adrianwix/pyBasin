"""
Basin stability estimation for the damped driven pendulum using Attractors.jl

The pendulum dynamics are:
    dθ/dt = θ̇
    dθ̇/dt = -α·θ̇ + T - K·sin(θ)

This mirrors the Python implementation in case_studies/pendulum/
"""

using OrdinaryDiffEq: DP5
using DynamicalSystems: CoupledODEs, AttractorsViaFeaturizing, GroupViaClustering,
    basins_fractions, statespace_sampler
using StaticArrays: SVector

# Pendulum parameters matching Python setup
const ALPHA = 0.1  # damping coefficient
const T_TORQUE = 0.5  # external torque (renamed to avoid conflict with Base.T)
const K = 1.0      # stiffness coefficient

function pendulum_rule(u, p, t)
    α, torque, k = p
    θ, θ̇ = u
    
    dθ = θ̇
    dθ̇ = -α * θ̇ + torque - k * sin(θ)
    return SVector(dθ, dθ̇)
end

function main()
    # Parameters
    p = [ALPHA, T_TORQUE, K]
    
    # Initial condition (will be varied)
    u0 = SVector(0.4, 0.0)
    
    # Create the dynamical system (DP5 = Dormand-Prince 5, same as diffrax.Dopri5 / MATLAB ode45)
    diffeq = (alg = DP5(), reltol = 1e-8, abstol = 1e-6, dt = 0.01)
    ds = CoupledODEs(pendulum_rule, u0, p; diffeq)
    
    # Define the state space grid for sampling
    θ_offset = asin(T_TORQUE / K)
    θ_range = range(-π + θ_offset, π + θ_offset, length=101)
    θ̇_range = range(-10.0, 10.0, length=101)
    grid = (θ_range, θ̇_range)
    
    # Feature function: extract mean angular velocity (distinguishes FP from LC)
    # Fixed Point: θ̇ → 0, Limit Cycle: θ̇ ≠ 0 (whirling)
    # Note: Ttr discards transient, T is kept. A is a StateSpaceSet.
    function featurizer(A, t)
        mean_θ̇ = sum(A[:, 2]) / size(A, 1)
        return SVector(abs(mean_θ̇))
    end
    
    # Use built-in clustering config
    # This actually uses the same algorithm than bSTAB Initialize a struct that contains instructions on how to group features in AttractorsViaFeaturizing. GroupViaClustering clusters features into groups using DBSCAN, similar to the original work by bSTAB (Stender and Hoffmann, 2021) and MCBB (Gelbrecht et al., 2020). Several options on clustering are available, see keywords below.
    grouping = GroupViaClustering(; min_neighbors=10, optimal_radius_method="silhouettes")
    
    # Create the featurizing mapper
    # T = integration time AFTER Ttr, so total integration = Ttr + T
    mapper = AttractorsViaFeaturizing(ds, featurizer, grouping;
        T = 100.0,   # integration time after transient (keep 100 time units)
        Ttr = 900.0, # transient time to discard
        Δt = 1.0,
    )
    
    # Sample initial conditions for basin stability estimation
    n_samples = 10000
    sampler, = statespace_sampler(grid)
    
    # Compute basin fractions (basin stability)
    println("Computing basin stability with $n_samples samples...")
    @time fractions = basins_fractions(mapper, sampler; N = n_samples, show_progress = true)
    
    println("\n=== Basin Stability Results ===")
    for (attractor_id, fraction) in fractions
        println("Attractor $attractor_id: $(round(fraction * 100, digits=2))%")
    end
    
    return fractions
end

# Run
fractions = main()
