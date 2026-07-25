    %%  Formation Control -- Controller Comparison with Diagnostics
%   Cooperative differential game theory - Jiang et al. (2020)
%
%   Compares Closed-Loop, Open-Loop, and Semi-Open-Loop controllers.
%   Uses correct TATD (RMS, all drones, per Jiang Eq.11) and
%   formation error measured against xd (not pairwise distances).
%
clc; clear all; close all;

%% ── Config ─────────────────  ───────────────────────────────────────────────
semiOpenTime = 5.0;     % Semi-open-loop replan interval (s)

%% ── Parameters & Riccati ─────────────────────────────────────────────────
P = sims.params();

fprintf('[1/4] Running Riccati solver...\n');
sims.riccatiSolver();
ric = load('riccati_solution.mat');

N       = P.N;
dt      = P.dt;
xd      = ric.xd;
n_steps = ric.n_steps;
t_vec   = (0 : n_steps-1) * dt;

%% ── Run Simulations ──────────────────────────────────────────────────────
fprintf('[2/4] Closed-loop...\n');
[x_cl,  u_cl]  = sims.closedLoop(P, ric);

fprintf('[3/4] Open-loop...\n');
[x_ol,  u_ol]  = sims.openLoop(P, ric);

fprintf('[4/4] Semi-open-loop (replan = %.1f s)...\n', semiOpenTime);
[x_sol, u_sol] = sims.semiOpenLoop(P, ric, semiOpenTime);

%% ── TATD (Jiang Eq. 11, RMS all drones, skip first 10s) ─────────────────
TATD_cl  = computeTATD(x_cl,  xd, N, n_steps, dt);
TATD_ol  = computeTATD(x_ol,  xd, N, n_steps, dt);
TATD_sol = computeTATD(x_sol, xd, N, n_steps, dt);

ol_pct  = ((TATD_ol  - TATD_cl) / TATD_cl) * 100;
sol_pct = ((TATD_sol - TATD_cl) / TATD_cl) * 100;

%% ── Formation Error vs Time (all drones vs xd) ───────────────────────────
ferr_cl  = computeFormationError(x_cl,  xd, N, n_steps);
ferr_ol  = computeFormationError(x_ol,  xd, N, n_steps);
ferr_sol = computeFormationError(x_sol, xd, N, n_steps);

%% ── Drone 1 Tracking Error vs Time ──────────────────────────────────────
pos_idx = [1, 4, 7];
err_cl  = vecnorm(x_cl(pos_idx,:)  - xd(:, pos_idx)', 2, 1);
err_ol  = vecnorm(x_ol(pos_idx,:)  - xd(:, pos_idx)', 2, 1);
err_sol = vecnorm(x_sol(pos_idx,:) - xd(:, pos_idx)', 2, 1);

%% ── Pre-Replan Drift (semi-open-loop) ────────────────────────────────────
replan_times = semiOpenTime : semiOpenTime : t_vec(end);
replan_ferr  = zeros(size(replan_times));
for r = 1:length(replan_times)
    k = find(t_vec >= replan_times(r), 1) - 1;
    if isempty(k) || k < 1, k = 1; end
    replan_ferr(r) = ferr_sol(k);
end

%% ── TATD Sensitivity: sweep measurement start time ──────────────────────
skip_sweep   = 0 : round(1/dt) : round(n_steps/2);
TATD_sw_cl   = zeros(size(skip_sweep));
TATD_sw_ol   = zeros(size(skip_sweep));
TATD_sw_sol  = zeros(size(skip_sweep));

for s = 1:length(skip_sweep)
    sk = skip_sweep(s);
    NT = n_steps - sk;
    if NT < 2, break; end
    TATD_sw_cl(s)  = computeTATD_fromSkip(x_cl,  xd, N, n_steps, dt, sk);
    TATD_sw_ol(s)  = computeTATD_fromSkip(x_ol,  xd, N, n_steps, dt, sk);
    TATD_sw_sol(s) = computeTATD_fromSkip(x_sol, xd, N, n_steps, dt, sk);
end

t_sweep   = skip_sweep * dt;
pct_sw_ol  = ((TATD_sw_ol  - TATD_sw_cl) ./ TATD_sw_cl) * 100;
pct_sw_sol = ((TATD_sw_sol - TATD_sw_cl) ./ TATD_sw_cl) * 100;

%% ── Steady-state window (last 20% of sim) ────────────────────────────────
ss_start     = round(0.8 * n_steps);
ss_ferr_cl   = mean(ferr_cl(ss_start:end));
ss_ferr_ol   = mean(ferr_ol(ss_start:end));
ss_ferr_sol  = mean(ferr_sol(ss_start:end));

%% ════════════════════════════════════════════════════════════════════════
%  TERMINAL REPORT
%  ════════════════════════════════════════════════════════════════════════
fprintf('\n════════════════════════════════════════════════════════════\n');
fprintf('  SIMULATION DIAGNOSTIC REPORT\n');
fprintf('════════════════════════════════════════════════════════════\n');
fprintf('  Semi-open-loop replan interval : %.1f s\n',  semiOpenTime);
fprintf('  TATD skip (first N seconds)    : 5 s  (hardcoded in computeTATD)\n');
fprintf('  Total simulation time          : %.2f s\n', t_vec(end));
fprintf('  Number of drones               : %d\n',     N);
fprintf('────────────────────────────────────────────────────────────\n');
fprintf('  %-20s  %10s  %14s\n', 'Controller', 'TATD (m)', '% vs Closed');
fprintf('  %-20s  %10.6f  %14s\n',   'Closed-Loop',    TATD_cl,  '--');
fprintf('  %-20s  %10.6f  %13.2f%%\n', 'Open-Loop',    TATD_ol,  ol_pct);
fprintf('  %-20s  %10.6f  %13.2f%%\n', 'Semi-Open-Loop', TATD_sol, sol_pct);
fprintf('────────────────────────────────────────────────────────────\n');
fprintf('  Steady-state formation error (last 20%% of sim):\n');
fprintf('    Closed-Loop    : %.5f m\n', ss_ferr_cl);
fprintf('    Open-Loop      : %.5f m\n', ss_ferr_ol);
fprintf('    Semi-Open-Loop : %.5f m\n', ss_ferr_sol);
fprintf('────────────────────────────────────────────────────────────\n');
fprintf('  Pre-replan formation error (semi-open-loop):\n');
fprintf('    %-10s  %s\n', 'Time (s)', 'Formation Error (m)');
for r = 1:length(replan_times)
    fprintf('    %-10.1f  %.5f m\n', replan_times(r), replan_ferr(r));
end
post_transient = replan_ferr(replan_ferr < 0.5);
fprintf('    Mean (excl. transient) : %.5f m\n', mean(post_transient));
fprintf('    Max  (excl. transient) : %.5f m\n', max(post_transient));
fprintf('════════════════════════════════════════════════════════════\n\n');

%% ── Sanity Checks ────────────────────────────────────────────────────────
fprintf('  SANITY CHECKS:\n');

% 1. TATD ordering
if TATD_cl <= TATD_sol && TATD_sol <= TATD_ol
    fprintf('  [PASS] TATD ordering correct: closed <= semi-open <= open\n');
else
    fprintf('  [FAIL] TATD ordering wrong: closed=%.5f  semi=%.5f  open=%.5f\n', ...
        TATD_cl, TATD_sol, TATD_ol);
end

% 2. Open-loop significantly worse
ratio = TATD_ol / TATD_cl;
if ratio > 5
    fprintf('  [PASS] Open-loop is significantly worse than closed-loop (%.1fx)\n', ratio);
else
    fprintf('  [WARN] Open-loop is only %.1fx closed-loop -- check sim length or drift\n', ratio);
end

% 3. Closed-loop steady-state formation error acceptable
if ss_ferr_cl < 0.15
    fprintf('  [PASS] Closed-loop steady-state formation error acceptable (%.4f m)\n', ss_ferr_cl);
else
    fprintf('  [WARN] Closed-loop steady-state formation error is %.4f m -- check cost weights\n', ss_ferr_cl);
end

% 4. Semi-open steady-state matches closed-loop
ss_ratio = ss_ferr_sol / ss_ferr_cl;
if ss_ratio < 3.0
    fprintf('  [PASS] Semi-open steady-state formation error close to closed-loop (%.2fx)\n', ss_ratio);
else
    fprintf('  [WARN] Semi-open steady-state is %.2fx closed-loop -- large replan interval?\n', ss_ratio);
end

% 5. Post-transient pre-replan drift is bounded
if mean(post_transient) < 0.2
    fprintf('  [PASS] Post-transient pre-replan drift bounded (mean = %.4f m)\n', mean(post_transient));
else
    fprintf('  [WARN] Post-transient pre-replan drift is high (mean = %.4f m)\n', mean(post_transient));
end

fprintf('\n');

%% ════════════════════════════════════════════════════════════════════════
%  PLOTS
%  ════════════════════════════════════════════════════════════════════════
c_cl  = [0.80 0.20 0.20];
c_ol  = [0.20 0.65 0.20];
c_sol = [0.20 0.40 0.80];
labels = {'Closed-Loop', 'Open-Loop', 'Semi-Open-Loop'};
t_skip_line = 10.0;   % matches hardcoded skip in computeTATD

%% Figure 1: TATD Bar Chart
figure('Name','Fig 1: TATD','Position',[50 600 500 380]);
TATDs = [TATD_cl, TATD_ol, TATD_sol];
b = bar(TATDs, 'FaceColor','flat');
b.CData = [c_cl; c_ol; c_sol];
set(gca, 'XTickLabel', labels, 'FontSize', 11);
ylabel('TATD (m)');
title('TATD Comparison (Jiang Eq. 11, skip first 10 s)');
grid on; box on;
for i = 1:3
    text(i, TATDs(i) + max(TATDs)*0.02, sprintf('%.5f m', TATDs(i)), ...
        'HorizontalAlignment','center','FontWeight','bold','FontSize',10);
end

%% Figure 2: Formation Error vs Time
figure('Name','Fig 2: Formation Error','Position',[560 600 900 400]);
plot(t_vec, ferr_cl,  'Color', c_cl,  'LineWidth', 1.8); hold on;
plot(t_vec, ferr_ol,  'Color', c_ol,  'LineWidth', 1.8);
plot(t_vec, ferr_sol, 'Color', c_sol, 'LineWidth', 1.8);

for r = 1:length(replan_times)
    xline(replan_times(r), '--', 'Color', [c_sol, 0.35], 'LineWidth', 0.8);
end
scatter(replan_times, replan_ferr, 50, c_sol, 'v', 'filled');
xline(t_skip_line, 'k--', 'LineWidth', 1.4, ...
    'Label', sprintf('TATD window start (%.0fs)', t_skip_line), ...
    'LabelVerticalAlignment','bottom');

xlabel('Time (s)'); ylabel('Mean Formation Error (m)');
title('Formation Error vs Time — All Controllers');
legend([labels, {'Replan event', 'Pre-replan sample'}], 'Location','northeast');
grid on; box on;

%% Figure 3: Drone 1 Tracking Error vs Time
figure('Name','Fig 3: Drone 1 Tracking Error','Position',[50 180 900 400]);
plot(t_vec, err_cl,  'Color', c_cl,  'LineWidth', 1.8); hold on;
plot(t_vec, err_ol,  'Color', c_ol,  'LineWidth', 1.8);
plot(t_vec, err_sol, 'Color', c_sol, 'LineWidth', 1.8);
xline(t_skip_line, 'k--', 'LineWidth', 1.4, ...
    'Label', sprintf('TATD window start (%.0fs)', t_skip_line), ...
    'LabelVerticalAlignment','bottom');
xlabel('Time (s)'); ylabel('Position Error Magnitude (m)');
title('Drone 1 Tracking Error vs Time');
legend(labels, 'Location','northeast');
grid on; box on;

%% Figure 4: TATD Sensitivity to Measurement Start Time
figure('Name','Fig 4: TATD Sensitivity','Position',[560 180 900 420]);
subplot(2,1,1);
plot(t_sweep, TATD_sw_cl,  'Color', c_cl,  'LineWidth', 1.8); hold on;
plot(t_sweep, TATD_sw_ol,  'Color', c_ol,  'LineWidth', 1.8);
plot(t_sweep, TATD_sw_sol, 'Color', c_sol, 'LineWidth', 1.8);
xline(t_skip_line, 'k--', 'LineWidth', 1.2, 'Label', 'Current skip');
ylabel('TATD (m)');
title('TATD vs Measurement Start Time');
legend(labels, 'Location','northeast');
grid on; box on;

subplot(2,1,2);
plot(t_sweep, pct_sw_ol,  'Color', c_ol,  'LineWidth', 1.8); hold on;
plot(t_sweep, pct_sw_sol, 'Color', c_sol, 'LineWidth', 1.8);
xline(t_skip_line, 'k--', 'LineWidth', 1.2, 'Label', 'Current skip');
yline(0, 'k-', 'LineWidth', 0.8);
xlabel('Measurement start time (s)');
ylabel('% increase vs closed-loop');
title('TATD % Increase vs Closed-Loop — Effect of Start Time');
legend({'Open-Loop', 'Semi-Open-Loop'}, 'Location','northeast');
grid on; box on;

%% Figure 5: Pre-Replan Drift Bar Chart
figure('Name','Fig 5: Pre-Replan Drift','Position',[50 180 650 380]);
bar(1:length(replan_times), replan_ferr, 'FaceColor', c_sol);
xlabel('Replan cycle');
ylabel('Formation error before replan (m)');
title(sprintf('Pre-Replan Drift Accumulation  (T_s = %.1f s)', semiOpenTime));
xticks(1:length(replan_times));
xticklabels(arrayfun(@(t) sprintf('%.0fs', t), replan_times, 'UniformOutput', false));
yline(mean(replan_ferr), 'r--', 'LineWidth', 1.5, 'Label', 'Mean (all)');
yline(mean(post_transient), 'b--', 'LineWidth', 1.5, 'Label', 'Mean (post-transient)');
grid on; box on;

%% Figure 6: XY Formation Overlay
figure('Name','Fig 6: XY Formation Overlay','Position',[100 100 1350 460]);
colors_d = lines(N);
c_dem    = [0.15 0.15 0.15];
data_x   = {x_cl, x_ol, x_sol};
titles_  = {'Closed-Loop', 'Open-Loop', 'Semi-Open-Loop'};

for p = 1:3
    subplot(1,3,p); hold on;
    X = data_x{p};
    plot(xd(:,1), xd(:,4), '--', 'Color', c_dem, 'LineWidth', 2);
    for i = 1:N
        xi = X((i-1)*10 + 1, :);
        yi = X((i-1)*10 + 4, :);
        plot(xi, yi, 'Color', colors_d(i,:), 'LineWidth', 1.2);
        scatter(xi(1),   yi(1),   50, colors_d(i,:), 'o', 'filled');
        scatter(xi(end), yi(end), 60, colors_d(i,:), 'd', 'filled');
    end
    title(titles_{p}); xlabel('x (m)'); ylabel('y (m)');
    axis equal; grid on;
    if p == 1
        legend('Desired','Drone paths','Start','End','Location','best');
    end
end
sgtitle('XY Formation Paths — All Controllers');

%% ── Save Figures ─────────────────────────────────────────────────────────
save_dir = '~/University/year4/semester1/EGH490-1/crazyflie_repo/MATLAB/plots/';

% Report figures -- names match \includegraphics in LaTeX
% Figures 3 and 5 are diagnostic only, saved with diag_ prefix
fig_map = { ...
    1, 'tatd_comparison';  ...   % TATD bar chart
    2, 'formation_error';  ...   % Formation error vs time
    3, 'diag_fig3';        ...   % Drone 1 tracking error (diagnostic)
    4, 'tatd_sensitivity'; ...   % TATD sensitivity sweep
    5, 'diag_fig5';        ...   % Pre-replan drift (diagnostic)
    6, 'xy_trajectory'     };    % XY formation paths

for row = 1:size(fig_map, 1)
    f     = fig_map{row, 1};
    name  = fig_map{row, 2};
    fpath = fullfile(save_dir, [name, '.png']);
    try
        exportgraphics(figure(f), fpath, 'Resolution', 300);
    catch
        saveas(figure(f), fpath);
    end
    fprintf('Saved: %s\n', fpath);
end
fprintf('All figures saved.\n');

%% ════════════════════════════════════════════════════════════════════════
%  LOCAL FUNCTIONS
%  ════════════════════════════════════════════════════════════════════════

function TATD = computeTATD(x, xd, N, n_steps, dt)
%COMPUTETATD  RMS tracking deviation per Jiang et al. (2020) Eq. 11.
%   Skips first 10 seconds to exclude initial transient.
    seconds = 5;
    skip    = round(seconds / dt);
    NT      = n_steps - skip;

    pos_idx = [];
    for i = 1:N
        base    = (i-1) * 10;
        pos_idx = [pos_idx, base+1, base+4, base+7];
    end

    err  = x(pos_idx, skip:end) - xd(skip:end, pos_idx)';
    TATD = sqrt(sum(err(:).^2) / (N * NT));
end

function TATD = computeTATD_fromSkip(x, xd, N, n_steps, dt, skip)
%COMPUTETATD_FROMSKIP  TATD with configurable skip index (for sweep).
    NT = n_steps - skip;
    if NT < 2, TATD = NaN; return; end

    pos_idx = [];
    for i = 1:N
        base    = (i-1) * 10;
        pos_idx = [pos_idx, base+1, base+4, base+7];
    end

    err  = x(pos_idx, skip+1:end) - xd(skip+1:end, pos_idx)';
    TATD = sqrt(sum(err(:).^2) / (N * NT));
end

function ferr = computeFormationError(X, xd, N, n_steps)
%COMPUTEFORMATIONERROR  Mean position error across all drones vs xd.
%   xd already encodes formation offsets per drone via pathManager.
%   Near-zero in steady state for closed/semi-open; growing for open-loop.
    ferr = zeros(1, n_steps);
    for k = 1:n_steps
        err_sum = 0;
        for i = 1:N
            pos_idx  = [(i-1)*10+1, (i-1)*10+4, (i-1)*10+7];
            actual   = X(pos_idx, k);
            desired  = xd(k, pos_idx)';
            err_sum  = err_sum + norm(actual - desired);
        end
        ferr(k) = err_sum / N;
    end
end