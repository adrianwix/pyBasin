"""
Basin stability scaling benchmark using Attractors.jl (DynamicalSystems.jl).

Benchmarks the AttractorsViaFeaturizing workflow for the damped driven pendulum
across different N values. Supports incremental runs: existing results are loaded
and only missing N values are benchmarked. Delete the output JSON to force a full re-run.

Pendulum dynamics:
    dθ/dt = θ̇
    dθ̇/dt = -α·θ̇ + T - K·sin(θ)

Run with:
    julia --project=benchmarks/end_to_end benchmarks/end_to_end/benchmark_julia_attractors.jl
"""

using OrdinaryDiffEq: DP5
using DynamicalSystems: CoupledODEs, AttractorsViaFeaturizing, GroupViaClustering,
    basins_fractions, statespace_sampler, HRectangle
using StaticArrays: SVector
using Statistics: mean, std
using JSON

# --- Configuration ---

const N_VALUES = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000]
const NUM_ROUNDS = 5

const ALPHA = 0.1
const T_TORQUE = 0.5
const K = 1.0

const RESULTS_PATH = joinpath(@__DIR__, "results",
    "julia_attractors_basin_stability_estimator_scaling.json")

# --- ODE ---

function pendulum_rule(u, p, t)
    α, torque, k = p
    θ, θ̇ = u
    dθ = θ̇
    dθ̇ = -α * θ̇ + torque - k * sin(θ)
    return SVector(dθ, dθ̇)
end

# --- Sampler ---

function make_sampler(seed::Int)
    θ_offset = asin(T_TORQUE / K)
    lb = [-π + θ_offset, -10.0]
    ub = [π + θ_offset, 10.0]
    sampler, = statespace_sampler(HRectangle(lb, ub), seed)
    return sampler
end

# --- Incremental merge ---

function load_existing_results()::Dict
    if isfile(RESULTS_PATH)
        return JSON.parsefile(RESULTS_PATH)
    end
    return Dict("num_rounds" => NUM_ROUNDS, "benchmarks" => [])
end

function get_completed_n_values(existing::Dict)::Set{Int}
    return Set{Int}(bench["N"] for bench in existing["benchmarks"])
end

function save_results(existing::Dict, new_benchmarks::Vector{Dict})
    all_benchmarks = vcat(existing["benchmarks"], new_benchmarks)
    sort!(all_benchmarks, by=b -> b["N"])
    existing["benchmarks"] = all_benchmarks
    existing["num_rounds"] = NUM_ROUNDS

    mkpath(dirname(RESULTS_PATH))
    open(RESULTS_PATH, "w") do f
        JSON.print(f, existing, 2)
    end
    println("Saved to: $RESULTS_PATH")
end

function append_result(existing::Dict, result::Dict)
    push!(existing["benchmarks"], result)
    sort!(existing["benchmarks"], by=b -> b["N"])
    existing["num_rounds"] = NUM_ROUNDS

    mkpath(dirname(RESULTS_PATH))
    open(RESULTS_PATH, "w") do f
        JSON.print(f, existing, 2)
    end
    println("  Saved N=$(result["N"]) to: $RESULTS_PATH")
end

# --- Expected basin stability values (from main_pendulum_case1.json, N=10000) ---

const EXPECTED_FP = 0.152
const EXPECTED_LC = 0.848
const BS_TOLERANCE = 0.02  # ~5.5 standard errors at N=10000

function validate_fractions(fractions::Dict, n::Int)::Bool
    n < 10_000 && return true

    vals = sort(collect(values(fractions)))
    if length(vals) != 2
        println("  VALIDATION FAILED: Expected 2 attractors, got $(length(vals)): $fractions")
        return false
    end

    fp_ok = abs(vals[1] - EXPECTED_FP) <= BS_TOLERANCE
    lc_ok = abs(vals[2] - EXPECTED_LC) <= BS_TOLERANCE

    if !fp_ok || !lc_ok
        println("  VALIDATION FAILED for N=$n")
        println("    Expected: FP≈$EXPECTED_FP, LC≈$EXPECTED_LC (tolerance ±$BS_TOLERANCE)")
        println("    Got:      $(round.(vals, digits=4))")
        return false
    end

    println("  Validation passed: $(round.(vals, digits=4)) ✓")
    return true
end

# --- Benchmark ---

function run_benchmark(n::Int)::Union{Dict, Nothing}
    p = [ALPHA, T_TORQUE, K]
    u0 = SVector(0.4, 0.0)
    diffeq = (alg = DP5(), reltol = 1e-8, abstol = 1e-6, dt = 0.01)

    function featurizer(A, t)
        mean_θ̇ = sum(A[:, 2]) / size(A, 1)
        return SVector(abs(mean_θ̇))
    end

    # max_used_features caps silhouettes computation to O(max_used_features²)
    # instead of O(N²), keeping each round fast at large N
    grouping = GroupViaClustering(; min_neighbors=10, optimal_radius_method="silhouettes", max_used_features=500)

    round_times = Float64[]

    for round_idx in 1:NUM_ROUNDS
        # Recreate ds and mapper each round: the mapper accumulates all feature
        # vectors internally, so reusing it across rounds inflates the feature
        # store to round_idx*N entries and causes OOM at large N.
        ds = CoupledODEs(pendulum_rule, u0, p; diffeq)
        mapper = AttractorsViaFeaturizing(ds, featurizer, grouping;
            T = 100.0,
            Ttr = 900.0,
            Δt = 1.0,
            threaded = true,
        )

        sampler = make_sampler(round_idx)
        t_start = time_ns()
        fractions = basins_fractions(mapper, sampler; N = n, show_progress = false)
        elapsed = (time_ns() - t_start) / 1e9
        push!(round_times, elapsed)
        println("  Round $round_idx/$NUM_ROUNDS: $(round(elapsed, digits=3))s")

        if round_idx == 1 && !validate_fractions(fractions, n)
            return nothing
        end
    end

    return Dict(
        "N" => n,
        "round_times" => round_times,
        "mean_time" => mean(round_times),
        "std_time" => std(round_times),
        "min_time" => minimum(round_times),
        "max_time" => maximum(round_times),
    )
end

function main()
    existing = load_existing_results()
    completed = get_completed_n_values(existing)
    pending = sort([n for n in N_VALUES if !(n in completed)])

    if isempty(pending)
        println("All N values already benchmarked. Delete $RESULTS_PATH to re-run.")
        return
    end

    println("Already completed: $(sort(collect(completed)))")
    println("Pending N values: $pending")

    # Warmup run (JIT compilation) - excluded from timing
    println("\nWarmup run (N=50, excluded from timing)...")
    p = [ALPHA, T_TORQUE, K]
    u0 = SVector(0.4, 0.0)
    diffeq = (alg = DP5(), reltol = 1e-8, abstol = 1e-6, dt = 0.01)
    ds = CoupledODEs(pendulum_rule, u0, p; diffeq)

    function featurizer(A, t)
        mean_θ̇ = sum(A[:, 2]) / size(A, 1)
        return SVector(abs(mean_θ̇))
    end

    grouping = GroupViaClustering(; min_neighbors=10, optimal_radius_method="silhouettes", max_used_features=500)
    mapper = AttractorsViaFeaturizing(ds, featurizer, grouping; T=100.0, Ttr=900.0, Δt=1.0, threaded=true)
    warmup_sampler = make_sampler(0)
    basins_fractions(mapper, warmup_sampler; N=50, show_progress=false)
    println("Warmup complete.\n")

    for n in pending
        println("Benchmarking N=$n ($NUM_ROUNDS rounds)...")
        result = run_benchmark(n)
        if result === nothing
            println("  Stopping benchmark due to validation failure at N=$n.")
            println("  Check featurizer / grouping parameters and re-run.")
            return
        end
        println("  Mean: $(round(result["mean_time"], digits=3))s ± $(round(result["std_time"], digits=3))s")
        append_result(existing, result)
        println()
    end
end

main()
