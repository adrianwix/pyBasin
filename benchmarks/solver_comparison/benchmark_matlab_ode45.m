% benchmark_matlab_ode45.m
% Benchmark MATLAB ode45 solver for damped driven pendulum ODE integration.
% Companion to Python benchmark_solver_comparison.py for cross-language comparison.
%
% Benchmarks ONLY raw ODE integration (no feature extraction or classification).
% Runs 5 rounds per N value with warmup, matching pytest-benchmark behavior.
%
% Run with:
%     matlab -nodisplay -nosplash -r "run('benchmark_matlab_ode45.m'); exit"

clear; clc; close all;

fprintf('========================================\n');
fprintf('MATLAB ODE45 Solver Benchmark\n');
fprintf('========================================\n');

%% Configuration (matching Python benchmark)
ALPHA = 0.1;
T = 0.5;
K = 1.0;

TIME_SPAN = [0, 1000];
RTOL = 1e-8;
ATOL = 1e-6;

MIN_LIMITS = [-pi + asin(T / K), -10.0];
MAX_LIMITS = [pi + asin(T / K), 10.0];

N_VALUES = [100000];
NUM_ROUNDS = 5;

options = odeset('RelTol', RTOL, 'AbsTol', ATOL);

%% Shutdown any existing parallel pool to ensure clean warmup timing
poolobj = gcp('nocreate');
if ~isempty(poolobj)
    delete(poolobj);
    fprintf('Shut down existing parallel pool.\n');
end

%% Warmup run to initialize parallel pool
fprintf('\nPerforming warmup run to initialize parallel pool...\n');
WARMUP_N = 100;

rng(42);
ic_warmup = zeros(WARMUP_N, 2);
ic_warmup(:, 1) = MIN_LIMITS(1) + (MAX_LIMITS(1) - MIN_LIMITS(1)) * rand(WARMUP_N, 1);
ic_warmup(:, 2) = MIN_LIMITS(2) + (MAX_LIMITS(2) - MIN_LIMITS(2)) * rand(WARMUP_N, 1);

all_Y_warmup = cell(WARMUP_N, 1);

tic;
parfor i = 1:WARMUP_N
    [~, ~] = ode45(@(t, y) ode_pendulum(t, y, ALPHA, T, K), TIME_SPAN, ic_warmup(i,:), options);
end
warmup_time = toc;
fprintf('Warmup complete in %.2f seconds.\n\n', warmup_time);

% Clear warmup arrays to free memory
clear ic_warmup;

%% Get parallel pool info
pool = gcp('nocreate');
fprintf('Using %d parallel workers\n\n', pool.NumWorkers);

%% Results storage
results = struct();
results.solver = 'matlab_ode45';
results.device = 'cpu';
results.num_rounds = NUM_ROUNDS;
results.benchmarks = [];

%% Run benchmarks for each N value
for n_idx = 1:length(N_VALUES)
    N = N_VALUES(n_idx);
    fprintf('Running benchmark for N = %d (%d/%d) with %d rounds...\n', ...
            N, n_idx, length(N_VALUES), NUM_ROUNDS);
    
    round_times = zeros(1, NUM_ROUNDS);
    
    for round = 1:NUM_ROUNDS
        fprintf('  Round %d/%d... ', round, NUM_ROUNDS);
        
        % Generate random initial conditions OUTSIDE timing
        rng(42 + round);
        ic_grid = zeros(N, 2);
        ic_grid(:, 1) = MIN_LIMITS(1) + (MAX_LIMITS(1) - MIN_LIMITS(1)) * rand(N, 1);
        ic_grid(:, 2) = MIN_LIMITS(2) + (MAX_LIMITS(2) - MIN_LIMITS(2)) * rand(N, 1);
        
        % Run benchmark - ONLY time ODE integration
        % Don't store full trajectories to avoid memory exhaustion
        tic_start = tic;
        
        parfor i = 1:N
            [~, ~] = ode45(@(t, y) ode_pendulum(t, y, ALPHA, T, K), TIME_SPAN, ic_grid(i,:), options);
        end
        
        elapsed_time = toc(tic_start);
        round_times(round) = elapsed_time;
        
        fprintf('%.2f seconds\n', elapsed_time);
        
        % Clear initial conditions array
        clear ic_grid;
    end
    
    % Compute statistics
    benchmark_result = struct();
    benchmark_result.n = N;
    benchmark_result.round_times = round_times;
    benchmark_result.mean_time = mean(round_times);
    benchmark_result.std_time = std(round_times);
    benchmark_result.min_time = min(round_times);
    benchmark_result.max_time = max(round_times);
    
    results.benchmarks = [results.benchmarks; benchmark_result];
    
    fprintf('  Summary: mean=%.2f±%.2f s, min=%.2f s, max=%.2f s\n\n', ...
            benchmark_result.mean_time, benchmark_result.std_time, ...
            benchmark_result.min_time, benchmark_result.max_time);
end

%% Save results to JSON
fprintf('========================================\n');
fprintf('Results Summary\n');
fprintf('========================================\n');

for i = 1:length(results.benchmarks)
    r = results.benchmarks(i);
    fprintf('N=%6d: mean=%.2f±%.2f s, rounds=%s\n', ...
            r.n, r.mean_time, r.std_time, mat2str(r.round_times, 2));
end

% Save to JSON file
results_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

json_file = fullfile(results_dir, 'matlab_benchmark_results.json');
json_str = jsonencode(results, 'PrettyPrint', true);
fid = fopen(json_file, 'w');
fprintf(fid, '%s', json_str);
fclose(fid);
fprintf('\nResults saved to: %s\n', json_file);

fprintf('\nBenchmark complete.\n');


%% ODE Function
function dydt = ode_pendulum(~, y, alpha, T, K)
    % Damped driven pendulum ODE
    %   dy/dt = [omega, -alpha*omega + T - K*sin(theta)]
    
    theta = y(1);
    omega = y(2);
    
    dtheta_dt = omega;
    domega_dt = -alpha * omega + T - K * sin(theta);
    
    dydt = [dtheta_dt; domega_dt];
end
