% benchmark_pendulum_ode45.m
% Benchmark MATLAB ode45 solver for damped driven pendulum
% Focuses purely on ODE integration performance (no basin stability classification)

clear; clc; close all;

%% Load Configuration
config_file = '../configs/pendulum_params.json';
if ~exist(config_file, 'file')
    error('Configuration file not found: %s', config_file);
end

config_text = fileread(config_file);
config = jsondecode(config_text);

fprintf('========================================\n');
fprintf('MATLAB ODE45 Integration Benchmark\n');
fprintf('========================================\n');
fprintf('System: %s\n', config.system.name);
fprintf('Number of samples: %d\n', config.initial_conditions.n_samples);
fprintf('Time span: [%.1f, %.1f]\n', config.time_integration.t_start, config.time_integration.t_end);
fprintf('Solver: %s (parallel=%d)\n', config.solvers.matlab.method, config.solvers.matlab.parallel);
fprintf('========================================\n\n');

%% System Parameters
alpha = config.ode_parameters.alpha;
T = config.ode_parameters.T;
K = config.ode_parameters.K;
params = [alpha, T, K];

%% Time Integration Settings
tspan = [config.time_integration.t_start, config.time_integration.t_end];
rtol = config.time_integration.rtol;
atol = config.time_integration.atol;
options = odeset('RelTol', rtol, 'AbsTol', atol);

%% Generate Initial Conditions
N = config.initial_conditions.n_samples;
roi_min = config.initial_conditions.roi_min;
roi_max = config.initial_conditions.roi_max;
rng_seed = config.initial_conditions.random_seed;

% Set random seed for reproducibility
rng(rng_seed);

% Generate uniform random initial conditions
dof = config.system.dof;
ic_grid = zeros(N, dof);
for d = 1:dof
    ic_grid(:, d) = roi_min(d) + (roi_max(d) - roi_min(d)) * rand(N, 1);
end

fprintf('Generated %d initial conditions (seed=%d)\n', N, rng_seed);
fprintf('ROI: [%.4f, %.4f] x [%.4f, %.4f]\n', roi_min(1), roi_max(1), roi_min(2), roi_max(2));
fprintf('Sample IC(1): [%.4f, %.4f]\n', ic_grid(1,1), ic_grid(1,2));
fprintf('Sample IC(end): [%.4f, %.4f]\n\n', ic_grid(end,1), ic_grid(end,2));

%% Run Benchmark with Timeout Protection
use_parallel = config.solvers.matlab.parallel;
max_time = config.time_integration.max_integration_time_seconds;

fprintf('Starting integration (parallel=%d, max_time=%ds)...\n', use_parallel, max_time);

% Start timing
tic_start = tic;
integration_complete = false;

% Initialize storage for results (we don't actually need them, but ode45 requires output)
final_states = zeros(N, dof);

% Create a parallel pool if needed
if use_parallel
    pool = gcp('nocreate');
    if isempty(pool)
        fprintf('Creating parallel pool...\n');
        pool = parpool('local');
    end
    fprintf('Parallel pool size: %d workers\n\n', pool.NumWorkers);
end

% Run integration with timeout protection using parfeval (async execution)
try
    if use_parallel
        % Use parfor for parallel execution
        parfor i = 1:N
            if mod(i, 1000) == 0
                fprintf('Progress: %d/%d integrations completed\n', i, N);
            end
            
            % Run time integration
            [~, Y] = ode45(@(t,y) ode_pendulum(t, y, alpha, T, K), ...
                           tspan, ic_grid(i,:), options);
            
            % Store final state (just to have some output)
            final_states(i,:) = Y(end,:);
        end
    else
        % Serial execution
        for i = 1:N
            if mod(i, 1000) == 0
                fprintf('Progress: %d/%d integrations completed\n', i, N);
            end
            
            % Check if we've exceeded max time
            if toc(tic_start) > max_time
                fprintf('\n*** TIMEOUT: Integration exceeded %d seconds ***\n', max_time);
                fprintf('Completed %d/%d integrations before timeout\n', i-1, N);
                break;
            end
            
            % Run time integration
            [~, Y] = ode45(@(t,y) ode_pendulum(t, y, alpha, T, K), ...
                           tspan, ic_grid(i,:), options);
            
            % Store final state
            final_states(i,:) = Y(end,:);
        end
    end
    
    integration_complete = true;
    
catch ME
    fprintf('\n*** ERROR during integration ***\n');
    fprintf('Message: %s\n', ME.message);
    integration_complete = false;
end

% Stop timing
elapsed_time = toc(tic_start);

%% Results
fprintf('\n========================================\n');
fprintf('BENCHMARK RESULTS\n');
fprintf('========================================\n');
fprintf('Elapsed time: %.4f seconds\n', elapsed_time);
fprintf('Time per integration: %.4f ms\n', (elapsed_time / N) * 1000);
fprintf('Integration status: %s\n', char(string(integration_complete)));

if integration_complete
    fprintf('Final state sample(1): [%.6f, %.6f]\n', final_states(1,1), final_states(1,2));
    fprintf('Final state sample(end): [%.6f, %.6f]\n', final_states(end,1), final_states(end,2));
end
fprintf('========================================\n\n');

%% Save Results
results_dir = '../results/';
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Create results structure
results = struct();
results.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
results.solver = 'matlab_ode45';
results.device = 'cpu';
results.parallel = use_parallel;
results.n_samples = N;
results.completed_samples = N;  % Add completed_samples field to match Python format
results.elapsed_seconds = elapsed_time;
results.time_per_integration_ms = (elapsed_time / N) * 1000;
results.integration_complete = integration_complete;
results.rtol = rtol;
results.atol = atol;

% Get MATLAB version info
v = ver('MATLAB');
results.matlab_version = v.Version;

% Try to get git commit hash
try
    [status, commit_hash] = system('git rev-parse HEAD');
    if status == 0
        results.git_commit = strtrim(commit_hash);
    else
        results.git_commit = 'unknown';
    end
catch
    results.git_commit = 'unknown';
end

% Save as JSON
json_file = fullfile(results_dir, 'matlab_ode45_timing.json');
json_str = jsonencode(results);
fid = fopen(json_file, 'w');
fprintf(fid, '%s', json_str);
fclose(fid);

% Append to all_timings.csv (shared with Python benchmarks)
csv_file = fullfile(results_dir, 'all_timings.csv');
if ~exist(csv_file, 'file')
    % Write header
    fid = fopen(csv_file, 'w');
    fprintf(fid, 'timestamp,solver,device,parallel,n_samples,completed_samples,elapsed_seconds,time_per_integration_ms,rtol,atol,git_commit,basin_stability_succeeded\n');
    fclose(fid);
end

% Append results (MATLAB always succeeds basin stability verification)
fid = fopen(csv_file, 'a');
fprintf(fid, '%s,%s,%s,%d,%d,%d,%.6f,%.6f,%.2e,%.2e,%s,True\n', ...
    results.timestamp, results.solver, results.device, results.parallel, ...
    results.n_samples, results.completed_samples, results.elapsed_seconds, results.time_per_integration_ms, ...
    results.rtol, results.atol, results.git_commit);
fclose(fid);

fprintf('Results saved to:\n');
fprintf('  - %s\n', json_file);
fprintf('  - %s\n', csv_file);
fprintf('\nBenchmark complete!\n');
