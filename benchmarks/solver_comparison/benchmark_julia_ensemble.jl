"""
Basin stability scaling benchmark using DifferentialEquations.jl EnsembleProblem.

Benchmarks CPU (EnsembleThreads + DP5, Float64) and GPU (EnsembleGPUKernel + GPUTsit5,
Float32) workflows for the damped driven pendulum across different N values.

Classification uses the same heuristic as PendulumFeatureExtractor:
if |max(ω) - mean(ω)| < 0.01 in steady state (t ≥ 950) → Fixed Point, else Limit Cycle.

Supports incremental runs: existing results are loaded and only missing N values
are benchmarked. Delete the output JSON to force a full re-run.

Pendulum dynamics:
    dθ/dt = ω
    dω/dt = -α·ω + T - K·sin(θ)

Run with:
    julia --threads=auto --project=benchmarks/solver_comparison benchmarks/solver_comparison/benchmark_julia_ensemble.jl [cpu|gpu|both]
"""

using OrdinaryDiffEq: DP5, ODEProblem, solve, remake, EnsembleProblem, EnsembleThreads
using StaticArrays: SVector
using Statistics: mean, std
using JSON
using Random: MersenneTwister
using DiffEqGPU: EnsembleGPUKernel, GPUTsit5
using CUDA

# --- Mode selection ---

const MODE = length(ARGS) > 0 ? lowercase(ARGS[1]) : "both"
const RUN_CPU = MODE in ("cpu", "both")
const RUN_GPU = MODE in ("gpu", "both")

# --- Configuration ---

const N_VALUES = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
const NUM_ROUNDS = 5

const ALPHA = 0.1
const T_TORQUE = 0.5
const K = 1.0

const TIME_SPAN = (0.0, 1000.0)
const N_SAVE = 10000
const SAVE_TIMES = range(TIME_SPAN[1], TIME_SPAN[2], length=N_SAVE)
const STEADY_STATE_T = 950.0
const RTOL = 1e-8
const ATOL = 1e-6
const FP_THRESHOLD = 0.01

const θ_OFFSET = asin(T_TORQUE / K)
const IC_LB = SVector(-π + θ_OFFSET, -10.0)
const IC_UB = SVector(π + θ_OFFSET, 10.0)

const RESULTS_PATH = joinpath(@__DIR__, "results",
    "julia_ensemble_basin_stability_scaling.json")
const GPU_RESULTS_PATH = joinpath(@__DIR__, "results",
    "julia_ensemble_gpu_basin_stability_scaling.json")

# GPU: save every 1s (~1001 points) — fewer than CPU to manage GPU memory at large N
const GPU_SAVEAT = 1.0f0

# --- Expected basin stability values (from main_pendulum_case1.json, N=10000) ---

const EXPECTED_FP = 0.152
const EXPECTED_LC = 0.848
const BS_TOLERANCE = 0.02

# --- ODE (out-of-place, SVector for performance) ---

function pendulum_rule(u, p, t)
    α, torque, k = p
    θ, ω = u
    dθ = ω
    dω = -α * ω + torque - k * sin(θ)
    return SVector(dθ, dω)
end

# --- Initial condition sampling ---

function sample_ics(n::Int, seed::Int)::Matrix{Float64}
    rng = MersenneTwister(seed)
    ics = Matrix{Float64}(undef, 2, n)
    for i in 1:n
        ics[1, i] = IC_LB[1] + (IC_UB[1] - IC_LB[1]) * rand(rng)
        ics[2, i] = IC_LB[2] + (IC_UB[2] - IC_LB[2]) * rand(rng)
    end
    return ics
end

# --- Classification (matches PendulumFeatureExtractor heuristic) ---
#
# From pendulum_feature_extractor.py:
#   y_1 = angular velocity (state index 1, 0-based → index 2 in Julia)
#   delta = |max(y_1) - mean(y_1)|
#   FP if delta < 0.01, else LC

function classify_solution(sol)::Int
    steady_idx = findfirst(t -> t >= STEADY_STATE_T, sol.t)
    u = sol.u
    ω_max = -Inf
    ω_sum = 0.0
    n = length(u) - steady_idx + 1
    for i in steady_idx:length(u)
        ω = u[i][2]
        ω_sum += ω
        if ω > ω_max
            ω_max = ω
        end
    end
    delta = abs(ω_max - ω_sum / n)
    return delta < FP_THRESHOLD ? 1 : 2  # 1=FP, 2=LC
end

# --- Basin stability from labels ---

function compute_fractions(labels::Vector{Int})::Dict{String,Float64}
    n = length(labels)
    fp_count = count(==(1), labels)
    lc_count = count(==(2), labels)
    return Dict("FP" => fp_count / n, "LC" => lc_count / n)
end

# --- Validation ---

function validate_fractions(fractions::Dict{String,Float64}, n::Int)::Bool
    n < 10_000 && return true

    fp = get(fractions, "FP", 0.0)
    lc = get(fractions, "LC", 0.0)
    fp_ok = abs(fp - EXPECTED_FP) <= BS_TOLERANCE
    lc_ok = abs(lc - EXPECTED_LC) <= BS_TOLERANCE

    if !fp_ok || !lc_ok
        println("  VALIDATION FAILED for N=$n")
        println("    Expected: FP≈$EXPECTED_FP, LC≈$EXPECTED_LC (tolerance ±$BS_TOLERANCE)")
        println("    Got:      FP=$(round(fp, digits=4)), LC=$(round(lc, digits=4))")
        return false
    end

    println("  Validation passed: FP=$(round(fp, digits=4)), LC=$(round(lc, digits=4)) ✓")
    return true
end

# --- Incremental JSON I/O ---

function load_existing_results(path::String)::Dict
    if isfile(path)
        return JSON.parsefile(path)
    end
    return Dict("num_rounds" => NUM_ROUNDS,
                "benchmarks" => [])
end

function get_completed_n_values(existing::Dict)::Set{Int}
    return Set{Int}(bench["N"] for bench in existing["benchmarks"])
end

function append_result(existing::Dict, result::Dict, path::String)
    push!(existing["benchmarks"], result)
    sort!(existing["benchmarks"], by=b -> b["N"])
    existing["num_rounds"] = NUM_ROUNDS

    mkpath(dirname(path))
    open(path, "w") do f
        JSON.print(f, existing, 2)
    end
    println("  Saved N=$(result["N"]) to: $path")
end

# --- CPU Benchmark ---

function run_cpu_benchmark(n::Int)::Union{Dict,Nothing}
    p = [ALPHA, T_TORQUE, K]
    u0 = SVector(0.0, 0.0)
    prob = ODEProblem(pendulum_rule, u0, TIME_SPAN, p)

    round_times = Float64[]
    validated = false

    for round_idx in 1:NUM_ROUNDS
        ics = sample_ics(n, round_idx)

        prob_func = let ics = ics
            (prob, i, repeat) -> remake(prob, u0 = SVector(ics[1, i], ics[2, i]))
        end

        output_func(sol, i) = (classify_solution(sol), false)

        ensemble_prob = EnsembleProblem(prob;
            prob_func = prob_func,
            output_func = output_func,
            safetycopy = false,
        )

        t_start = time_ns()
        sim = solve(ensemble_prob, DP5(), EnsembleThreads();
            trajectories = n,
            saveat = SAVE_TIMES,
            reltol = RTOL,
            abstol = ATOL,
        )
        elapsed = (time_ns() - t_start) / 1e9
        push!(round_times, elapsed)
        println("  Round $round_idx/$NUM_ROUNDS: $(round(elapsed, digits=3))s")

        if !validated && n >= 10_000
            labels = Int[sim.u[i] for i in 1:n]
            fractions = compute_fractions(labels)
            if !validate_fractions(fractions, n)
                return nothing
            end
            validated = true
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

# --- GPU Benchmark ---

function run_gpu_benchmark(n::Int)::Union{Dict,Nothing}
    p = SVector{3,Float32}(Float32(ALPHA), Float32(T_TORQUE), Float32(K))
    u0 = SVector{2,Float32}(0.0f0, 0.0f0)
    prob = ODEProblem{false}(pendulum_rule, u0, (0.0f0, 1000.0f0), p)

    round_times = Float64[]
    validated = false

    for round_idx in 1:NUM_ROUNDS
        ics = sample_ics(n, round_idx)

        prob_func = let ics = ics
            (prob, i, repeat) -> remake(prob,
                u0 = SVector{2,Float32}(Float32(ics[1, i]), Float32(ics[2, i])))
        end

        ensemble_prob = EnsembleProblem(prob;
            prob_func = prob_func,
            safetycopy = false,
        )

        t_start = time_ns()
        sim = CUDA.@sync solve(ensemble_prob, GPUTsit5(),
            EnsembleGPUKernel(CUDA.CUDABackend());
            trajectories = n,
            saveat = GPU_SAVEAT,
            adaptive = true,
            dt = 0.1f0,
            reltol = 1.0f-6,
            abstol = 1.0f-6,
        )
        labels = Int[classify_solution(sim[i]) for i in 1:n]
        elapsed = (time_ns() - t_start) / 1e9
        push!(round_times, elapsed)
        println("  Round $round_idx/$NUM_ROUNDS: $(round(elapsed, digits=3))s")

        if !validated && n >= 10_000
            fractions = compute_fractions(labels)
            if !validate_fractions(fractions, n)
                return nothing
            end
            validated = true
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

# --- Warmup ---

function warmup_cpu()
    println("CPU warmup (JIT compilation)...")
    n = 100
    p = [ALPHA, T_TORQUE, K]
    u0 = SVector(0.0, 0.0)
    prob = ODEProblem(pendulum_rule, u0, TIME_SPAN, p)
    ics = sample_ics(n, 0)
    pf(prob, i, repeat) = remake(prob, u0 = SVector(ics[1, i], ics[2, i]))
    of(sol, i) = (classify_solution(sol), false)
    ep = EnsembleProblem(prob; prob_func=pf, output_func=of, safetycopy=false)
    solve(ep, DP5(), EnsembleThreads();
        trajectories=n, saveat=SAVE_TIMES, reltol=RTOL, abstol=ATOL)
    println("CPU warmup complete.\n")
end

function warmup_gpu()
    println("GPU warmup (kernel compilation)...")
    n = 100
    p = SVector{3,Float32}(Float32(ALPHA), Float32(T_TORQUE), Float32(K))
    u0 = SVector{2,Float32}(0.0f0, 0.0f0)
    prob = ODEProblem{false}(pendulum_rule, u0, (0.0f0, 1000.0f0), p)
    ics = sample_ics(n, 0)
    pf(prob, i, repeat) = remake(prob,
        u0 = SVector{2,Float32}(Float32(ics[1, i]), Float32(ics[2, i])))
    ep = EnsembleProblem(prob; prob_func=pf, safetycopy=false)
    CUDA.@sync solve(ep, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend());
        trajectories=n, saveat=GPU_SAVEAT, adaptive=true,
        dt=0.1f0, reltol=1.0f-6, abstol=1.0f-6)
    println("GPU warmup complete.\n")
end

# --- Run benchmarks for one mode ---

function run_mode(mode_name::String, run_func::Function, results_path::String)
    existing = load_existing_results(results_path)
    completed = get_completed_n_values(existing)
    pending = sort([n for n in N_VALUES if !(n in completed)])

    if isempty(pending)
        println("All N values already benchmarked. Delete $(basename(results_path)) to re-run.")
        return
    end

    println("Already completed: $(sort(collect(completed)))")
    println("Pending N values: $pending\n")

    for n in pending
        println("Benchmarking N=$n ($NUM_ROUNDS rounds)...")
        result = run_func(n)
        if result === nothing
            println("  Stopping due to validation failure at N=$n.")
            return
        end
        println("  Mean: $(round(result["mean_time"], digits=3))s ± $(round(result["std_time"], digits=3))s\n")
        append_result(existing, result, results_path)
    end
end

# --- Main ---

function main()
    println("Julia EnsembleProblem Basin Stability Benchmark")
    println("Mode: $MODE")
    println("Threads: $(Threads.nthreads())")
    has_cuda = CUDA.functional()
    if has_cuda
        println("CUDA: $(CUDA.name(CUDA.device()))")
    else
        println("CUDA: not available")
    end
    println("================================================\n")

    if RUN_CPU
        warmup_cpu()
        println("--- CPU Benchmark (EnsembleThreads, DP5, Float64) ---\n")
        run_mode("CPU", run_cpu_benchmark, RESULTS_PATH)
    end

    if RUN_GPU
        if !has_cuda
            println("\nGPU mode requested but CUDA not available. Skipping.\n")
        else
            warmup_gpu()
            println("--- GPU Benchmark (EnsembleGPUKernel, GPUTsit5, Float32) ---\n")
            run_mode("GPU", run_gpu_benchmark, GPU_RESULTS_PATH)
        end
    end
end

main()
