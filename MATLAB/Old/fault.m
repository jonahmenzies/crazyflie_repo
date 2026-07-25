%%  Semi-Open-Loop Fault Tolerance Test
%   Cooperative differential game theory - Jiang et al. (2020)
%
%   Randomly drops 3 of the scheduled pings and runs N_trials with
%   different drop combinations to show TATD distribution.
%
%   Hardware indexing note:
%     When a ping arrives at time t_in, the correct c/e is selected as:
%       cN = ping_num + 1   (ping_num is 1-indexed: ping 1 = first ping at t=replan_interval)
%       load c_all{cN}, e_all{cN},  set tau = 0,  x0_plan = measured state
%     This makes the system robust to slightly irregular ping timing.
%
clc; clear all; close all;

%% Parameters
params;

E_N = eye(N);
A   = kron(E_N, Ai);
B   = kron(E_N, bi);

Ri     = cell(N, 1);
R_vals = [5600, 1005, 1050, 1050, 2920];
for i = 1:N
    Ri{i} = R_vals(i) * eye(4);
end

load('riccati_solution.mat', 't_s', 's_store', 't_p', 'p_store', ...
     'sum_BR', 'xd', 'tT', 'n_steps');

%% Replan Settings
replan_interval = 5.0;
replan_steps    = round(replan_interval/dt);

%% Pre-compute c and e (each solved to tT)
replan_starts = t0 : replan_interval : tT - dt;
n_intervals   = length(replan_starts);

t_c_all = cell(n_intervals, 1);
c_all   = cell(n_intervals, 1);
t_e_all = cell(n_intervals, 1);
e_all   = cell(n_intervals, 1);

fprintf('Pre-computing c/e for %d intervals (each to tT)...\n', n_intervals);
for r = 1:n_intervals
    t_start = replan_starts(r);
    [t_c_all{r}, c_all{r}, t_e_all{r}, e_all{r}] = ...
        utils.solveLocalODEs(t_start, tT, t_s, s_store, t_p, p_store, A, sum_BR, N);
end
fprintf('Pre-computation complete.\n\n');

%% TATD setup
skip    = round(5/dt);
NT      = n_steps - skip;
pos_idx = [];
for i = 1:N
    base    = (i-1)*10;
    pos_idx = [pos_idx, base+1, base+4, base+7];
end

%% Ping schedule
%  Pings are attempted at k = replan_steps, 2*replan_steps, ...
%  Ping index 1 = first ping (t = replan_interval), etc.
ping_steps = replan_steps : replan_steps : n_steps-1;   % step indices
n_pings    = length(ping_steps);
n_drops    = 3;                                          % pings to drop
N_trials   = 20;                                         % random drop combos

fprintf('Pings scheduled at: %s s\n', ...
        num2str((ping_steps-1)*dt, '%.0f '));
fprintf('Randomly dropping %d of %d pings per trial.\n\n', n_drops, n_pings);

%% Run trials
tatd_all    = zeros(N_trials, 1);
dropped_all = zeros(N_trials, n_drops);

for trial = 1:N_trials
    %-- Randomly choose which pings to drop ----------------------------
    dropped = sort(randperm(n_pings, n_drops));   % e.g. [2 4 5]
    dropped_all(trial,:) = dropped;

    x_sim   = zeros(10*N, n_steps);
    x_sim(:,1) = x0;
    x0_plan = x0;
    tau     = 0;
    r_idx   = 1;

    for k = 1:n_steps-1

        %-- Ping attempt -----------------------------------------------
        ping_num = find(ping_steps == k, 1);   % non-empty if ping scheduled
        if ~isempty(ping_num)
            if ~ismember(ping_num, dropped)
                %-- Packet received: use timestamp to snap to correct c/e
                r_idx = min(ping_num + 1, n_intervals);
                x0_plan = x_sim(:,k);
                tau     = 0;
            end
            % else: dropped -- tau keeps running, c/e stays valid
        end

        t_k   = (k-1)*dt;
        t_tau = tau*dt;

        s_k = utils.lookupMatrix(t_k, t_s, s_store);
        p_k = interp1(t_p, p_store, t_k, 'linear', 'extrap')';
        c_k = utils.lookupMatrix(t_tau, t_c_all{r_idx}, c_all{r_idx});
        e_k = interp1(t_e_all{r_idx}, e_all{r_idx}, t_tau, 'linear', 'extrap')';

        x_hat = c_k*x0_plan + e_k;
        u_k   = utils.computeControl(x_hat, s_k, p_k, B, Ri, alpha, N, m, g);
        x_sim(:,k+1) = x_sim(:,k) + dt*(A*x_sim(:,k) + B*u_k);

        tau = tau + 1;
    end

    err            = x_sim(pos_idx, skip:end) - xd(skip:end, pos_idx)';
    tatd_all(trial) = sqrt(sum(err(:).^2) / (N*NT));
    fprintf('  Trial %2d  dropped pings: [%s]  TATD = %.4f m\n', ...
            trial, num2str(dropped), tatd_all(trial));
end

%% Baseline: no drops
x_nom   = zeros(10*N, n_steps);
x_nom(:,1) = x0;
x0_plan = x0;
tau     = 0;
r_idx   = 1;
for k = 1:n_steps-1
    ping_num = find(ping_steps == k, 1);
    if ~isempty(ping_num)
        r_idx   = min(ping_num + 1, n_intervals);
        x0_plan = x_nom(:,k);
        tau     = 0;
    end
    t_k   = (k-1)*dt;
    t_tau = tau*dt;
    s_k   = utils.lookupMatrix(t_k, t_s, s_store);
    p_k   = interp1(t_p, p_store, t_k, 'linear', 'extrap')';
    c_k   = utils.lookupMatrix(t_tau, t_c_all{r_idx}, c_all{r_idx});
    e_k   = interp1(t_e_all{r_idx}, e_all{r_idx}, t_tau, 'linear', 'extrap')';
    x_hat = c_k*x0_plan + e_k;
    u_k   = utils.computeControl(x_hat, s_k, p_k, B, Ri, alpha, N, m, g);
    x_nom(:,k+1) = x_nom(:,k) + dt*(A*x_nom(:,k) + B*u_k);
    tau = tau + 1;
end
err_nom  = x_nom(pos_idx, skip:end) - xd(skip:end, pos_idx)';
TATD_nom = sqrt(sum(err_nom(:).^2) / (N*NT));

fprintf('\nNo-drop baseline TATD = %.4f m\n', TATD_nom);
fprintf('With 3 drops -- mean = %.4f m,  std = %.4f m,  max = %.4f m\n', ...
        mean(tatd_all), std(tatd_all), max(tatd_all));

%% Plot 1 - TATD distribution across trials
figure;
histogram(tatd_all, 8, 'FaceColor', [0.2 0.5 0.9], 'EdgeColor', 'w', ...
          'FaceAlpha', 0.8);
hold on;
xline(TATD_nom,        'g--', 'LineWidth', 2);
xline(mean(tatd_all),  'b-',  'LineWidth', 2);
xlabel('TATD (m)');
ylabel('Count');
title(sprintf('TATD distribution: %d trials, 3 random drops each', N_trials));
legend(sprintf('No-drop baseline (%.4f m)', TATD_nom), ...
       sprintf('Mean with drops (%.4f m)', mean(tatd_all)), ...
       'Location', 'northwest');
grid on;
saveas(gcf, '/home/jonah/Downloads/fault_test_hist.png');

%% Plot 2 - Drone 1 x trajectory: baseline vs worst/best drop trial
[~, worst] = max(tatd_all);
[~, best]  = min(tatd_all);

% Re-run worst and best to get trajectories
trials_to_plot = struct('idx', {worst, best}, ...
                        'label', {'Worst drop combo','Best drop combo'}, ...
                        'color', {[0.8 0.2 0.2], [0.2 0.7 0.3]});

t_vec = (0:n_steps-1)*dt;
figure; hold on;
plot(t_vec, xd(:,1), 'b--', 'LineWidth', 1.5);
plot(t_vec, x_nom(1,:), 'k-', 'LineWidth', 1.5);

for s = 1:2
    trial_idx = trials_to_plot(s).idx;
    dropped   = dropped_all(trial_idx,:);

    x_sim   = zeros(10*N, n_steps);
    x_sim(:,1) = x0;
    x0_plan = x0;
    tau     = 0;
    r_idx   = 1;

    for k = 1:n_steps-1
        ping_num = find(ping_steps == k, 1);
        if ~isempty(ping_num) && ~ismember(ping_num, dropped)
            r_idx   = min(ping_num + 1, n_intervals);
            x0_plan = x_sim(:,k);
            tau     = 0;
        end
        t_k   = (k-1)*dt;
        t_tau = tau*dt;
        s_k   = utils.lookupMatrix(t_k, t_s, s_store);
        p_k   = interp1(t_p, p_store, t_k, 'linear', 'extrap')';
        c_k   = utils.lookupMatrix(t_tau, t_c_all{r_idx}, c_all{r_idx});
        e_k   = interp1(t_e_all{r_idx}, e_all{r_idx}, t_tau, 'linear', 'extrap')';
        x_hat = c_k*x0_plan + e_k;
        u_k   = utils.computeControl(x_hat, s_k, p_k, B, Ri, alpha, N, m, g);
        x_sim(:,k+1) = x_sim(:,k) + dt*(A*x_sim(:,k) + B*u_k);
        tau = tau + 1;
    end

    plot(t_vec, x_sim(1,:), '-', 'Color', trials_to_plot(s).color, ...
         'LineWidth', 1);

    % Mark dropped pings
    for dp = dropped
        xline((ping_steps(dp)-1)*dt, '--', ...
              'Color', trials_to_plot(s).color, 'Alpha', 0.5, 'LineWidth', 0.8);
    end
end

xlabel('Time (s)'); ylabel('x (m)');
title('Drone 1 x: baseline vs worst/best 3-drop combination');
legend('Demanded', 'No drops', ...
       sprintf('Worst drops %s (%.4f m)', mat2str(dropped_all(worst,:)), tatd_all(worst)), ...
       sprintf('Best drops  %s (%.4f m)', mat2str(dropped_all(best,:)),  tatd_all(best)), ...
       'Location', 'best');
grid on;
saveas(gcf, '/home/jonah/Downloads/fault_test_traj.png');

