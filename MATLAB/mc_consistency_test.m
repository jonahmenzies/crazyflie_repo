%%  Monte Carlo Consistency Test — validate ssErr / TATD stability
%   Perturbs x0 and waypoints across n_trials runs, checks whether
%   ssErr and TATD-last-20% stay consistent across conditions.
clc; clear all; close all;

n_trials     = 15;
semiOpenTime = 5.0;
ss_frac      = 0.2;
x0_noise_std = 0.3;    % m, position perturbation stdev
wp_noise_std = 0.2;    % m, waypoint perturbation stdev

rng(1);   % reproducible

P_nom = sims.params();
x0_nom = P_nom.x0;
wp_nom = P_nom.waypoints;

ssErr_cl  = zeros(n_trials,1); ssErr_ol  = zeros(n_trials,1); ssErr_sol = zeros(n_trials,1);
TATD20_cl = zeros(n_trials,1); TATD20_ol = zeros(n_trials,1); TATD20_sol = zeros(n_trials,1);

for trial = 1:n_trials
    fprintf('── Trial %d/%d ──\n', trial, n_trials);

    x0_trial = x0_nom + x0_noise_std * randn(size(x0_nom));
    wp_trial = wp_nom + wp_noise_std * randn(size(wp_nom));

    P = sims.params(x0_trial, wp_trial);
    sims.riccatiSolver(P);        % <-- was sims.riccatiSolver()
    ric = load('riccati_solution.mat');

    N = P.N; dt = P.dt; xd = ric.xd; n_steps = ric.n_steps;

    [x_cl,  ~] = sims.closedLoop(P, ric);
    [x_ol,  ~] = sims.openLoop(P, ric);
    [x_sol, ~] = sims.semiOpenLoop(P, ric, semiOpenTime);

    ferr_cl  = computeFormationError(x_cl,  xd, N, n_steps);
    ferr_ol  = computeFormationError(x_ol,  xd, N, n_steps);
    ferr_sol = computeFormationError(x_sol, xd, N, n_steps);

    ss_start = round((1-ss_frac) * n_steps);
    ssErr_cl(trial)  = mean(ferr_cl(ss_start:end));
    ssErr_ol(trial)  = mean(ferr_ol(ss_start:end));
    ssErr_sol(trial) = mean(ferr_sol(ss_start:end));

    TATD20_cl(trial)  = computeTATD_lastFrac(x_cl,  xd, N, n_steps, dt, ss_frac);
    TATD20_ol(trial)  = computeTATD_lastFrac(x_ol,  xd, N, n_steps, dt, ss_frac);
    TATD20_sol(trial) = computeTATD_lastFrac(x_sol, xd, N, n_steps, dt, ss_frac);
end

%% ── Report ────────────────────────────────────────────────────────────
fprintf('\n════════════════════════════════════════════════════════════\n');
fprintf('  CONSISTENCY ACROSS %d RANDOMISED TRIALS\n', n_trials);
fprintf('════════════════════════════════════════════════════════════\n');
fprintf('  %-16s  %-22s  %-22s\n', 'Controller', 'ssErr: mean±std (CV%)', 'TATD(last20%%): mean±std (CV%%)');
report_row('Closed-Loop',    ssErr_cl,  TATD20_cl);
report_row('Open-Loop',      ssErr_ol,  TATD20_ol);
report_row('Semi-Open-Loop', ssErr_sol, TATD20_sol);
fprintf('════════════════════════════════════════════════════════════\n');
fprintf('  CV%% = coefficient of variation (std/mean * 100). Lower = more consistent.\n');

function report_row(name, ssvals, tvals)
    ss_cv = std(ssvals)/mean(ssvals)*100;
    t_cv  = std(tvals)/mean(tvals)*100;
    fprintf('  %-16s  %.4f±%.4f (%.1f%%)   %.4f±%.4f (%.1f%%)\n', ...
        name, mean(ssvals), std(ssvals), ss_cv, mean(tvals), std(tvals), t_cv);
end

function ferr = computeFormationError(X, xd, N, n_steps)
    ferr = zeros(1, n_steps);
    for k = 1:n_steps
        err_sum = 0;
        for i = 1:N
            pos_idx = [(i-1)*10+1, (i-1)*10+4, (i-1)*10+7];
            err_sum = err_sum + norm(X(pos_idx,k) - xd(k,pos_idx)');
        end
        ferr(k) = err_sum / N;
    end
end

function TATD = computeTATD_lastFrac(x, xd, N, n_steps, dt, frac)
%COMPUTETATD_LASTFRAC  TATD (RMS, Jiang Eq.11 style) over the last frac
%of the simulation only -- e.g. frac=0.2 for last 20%.
    start_k = round((1-frac) * n_steps) + 1;
    NT = n_steps - start_k + 1;
    pos_idx = [];
    for i = 1:N
        base = (i-1)*10;
        pos_idx = [pos_idx, base+1, base+4, base+7];
    end
    err = x(pos_idx, start_k:end) - xd(start_k:end, pos_idx)';
    TATD = sqrt(sum(err(:).^2) / (N * NT));
end