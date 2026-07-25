%%
clc; clear all; close all;

%% Parameters
params;

E_N = eye(N);
A   = kron(E_N, Ai);
B   = kron(E_N, bi);

%% Cost Matrices
Ri     = cell(N, 1);
R_vals = [5600, 1005, 1050, 1050, 2920];
for i = 1:N
    Ri{i} = R_vals(i) * eye(4);
end

%% Load Riccati Solution
load('riccati_solution.mat', 't_s', 's_store', 't_p', 'p_store', 'sum_BR', 'xd', 'tT', 'n_steps');

%% Open-Loop Estimation ODEs
fprintf('Solving open-loop ODEs...\n');
tspan_fwd = [t0, tT];

c0_vec = eye(10*N);
c0_vec = c0_vec(:);
[t_c, c_vec_ode] = ode45(@(t, cv) cODE(t, cv, t_s, s_store, A, sum_BR, N), tspan_fwd, c0_vec);

c_store = zeros(10*N, 10*N, length(t_c));
for k = 1:length(t_c)
    c_store(:,:,k) = reshape(c_vec_ode(k,:), 10*N, 10*N);
end

e0 = zeros(10*N, 1);
[t_e, e_vec_ode] = ode45(@(t, e) eODE(t, e, t_s, s_store, t_p, p_store, A, sum_BR, N), tspan_fwd, e0);

%% Semi-Open-Loop Simulation
fprintf('Running semi-open-loop simulation...\n');
replan_interval = 5.0;
replan_steps    = round(replan_interval / dt);

x_sol = zeros(10*N, n_steps);
x_sol(:,1) = x0;
u_sol = zeros(4*N, n_steps);

% Solve initial c and e from t0
fprintf('  Solving initial c/e ODEs...\n');
[t_c_loc, c_store_loc, t_e_loc, e_vec_loc] = solveLocalODEs(...
    t0, tT, t_s, s_store, t_p, p_store, A, sum_BR, N);

x0_plan = x0;
tau     = 0;

for k = 1:n_steps-1
    if mod(k-1, replan_steps) == 0 && k > 1
        t_replan = (k-1)*dt;
        x0_plan  = x_sol(:,k);
        tau      = 0;
        fprintf('  Replanning at t = %.1f s\n', t_replan);
        [t_c_loc, c_store_loc, t_e_loc, e_vec_loc] = solveLocalODEs(...
            t_replan, tT, t_s, s_store, t_p, p_store, A, sum_BR, N);
    end

    t_k   = (k-1)*dt;   % absolute — for s and p
    t_tau = tau * dt;   % phase    — for c and e (seconds into current horizon)

    s_k = lookupMatrix(t_k,   t_s, s_store);
    p_k = interp1(t_p, p_store, t_k,   'linear', 'extrap')';
    c_k = lookupMatrix(t_tau, t_c_loc, c_store_loc);
    e_k = interp1(t_e_loc, e_vec_loc, t_tau, 'linear', 'extrap')';

    x_hat = c_k*x0_plan + e_k;
    u_k   = computeControl(x_hat, s_k, p_k, B, Ri, alpha, N, m, g);
    u_sol(:,k)   = u_k;
    x_sol(:,k+1) = x_sol(:,k) + dt*(A*x_sol(:,k) + B*u_k);

    tau = tau + 1;
end

%% Performance (TATD)
skip = round(5/dt);
NT   = n_steps - skip;

pos_idx = [];
for i = 1:N
    base    = (i-1)*10;
    pos_idx = [pos_idx, base+1, base+4, base+7];
end

err_sol  = x_sol(pos_idx, skip:end) - xd(skip:end, pos_idx)';
TATD_sol = sqrt(sum(err_sol(:).^2) / (N*NT));

fprintf('Semi-Open-Loop TATD = %.6f metres\n', TATD_sol);

%% Plots
t_vec      = (0:n_steps-1)*dt;
colors     = lines(N);
drone_lbls = {'Drone 1','Drone 2','Drone 3','Drone 4','Drone 5'};

figure;
h1 = plot(t_vec, xd(:,1),    'b--', 'LineWidth', 1.5); hold on;
h2 = plot(t_vec, x_sol(1,:), 'g-',  'LineWidth', 1);
title('Drone 1 — Demanded vs Actual x');
xlabel('Time (s)'); ylabel('x (m)');
legend([h1,h2], 'Demanded','Semi-Open-Loop'); grid on;

figure;
subplot(2,1,1); hold on;
h_sol = gobjects(N,1);
for i = 1:N
    h_sol(i) = plot(t_vec, x_sol((i-1)*10+1,:), 'Color', colors(i,:));
end
title('Semi-Open-Loop x — All Drones');
xlabel('Time (s)'); ylabel('x (m)');
legend(h_sol, drone_lbls); grid on;

subplot(2,1,2); hold on;
h_u = gobjects(N,1);
for i = 1:N
    h_u(i) = plot(t_vec, u_sol((i-1)*4+3,:), 'Color', colors(i,:));
end
title('Semi-Open-Loop Thrust — All Drones');
xlabel('Time (s)'); ylabel('Thrust (N)');
legend(h_u, drone_lbls); grid on;

figure; hold on;
h_sol2 = gobjects(N,1);
for i = 1:N
    h_sol2(i) = plot(x_sol((i-1)*10+1,:), x_sol((i-1)*10+4,:), 'Color', colors(i,:));
    plot(x_sol((i-1)*10+1,1), x_sol((i-1)*10+4,1), 'o', ...
         'MarkerSize', 8, 'MarkerFaceColor', colors(i,:), 'Color', colors(i,:));
end
title('Semi-Open-Loop — x vs y');
xlabel('x (m)'); ylabel('y (m)'); grid on; axis equal;
legend(h_sol2, drone_lbls);

figure;
h3d_sol = plot3(x_sol(1,:), x_sol(4,:), x_sol(7,:), 'g-',  'LineWidth', 1.5); hold on;
h3d_d   = plot3(xd(:,1),    xd(:,4),    xd(:,7),    'b--', 'LineWidth', 1.5);
h3d_s   = plot3(x_sol(1,1), x_sol(4,1), x_sol(7,1), 'ko',  ...
                'MarkerSize', 10, 'MarkerFaceColor', 'k');
title('Drone 1 — 3D Trajectory');
xlabel('x (m)'); ylabel('y (m)'); zlabel('z (m)');
legend([h3d_sol,h3d_d,h3d_s], 'Semi-Open-Loop','Demanded','Start');
grid on; view(45, 30);

saveas(gcf, '/home/jonah/Downloads/plot_sol.png');

%% -----------------------------------------------------------------------
%  ODE Right-Hand Sides
%% -----------------------------------------------------------------------

function dcdt = cODE(t, cv, t_s, s_store, A, sum_BR, N)
    S    = lookupMatrix(t, t_s, s_store);
    C    = reshape(cv, 10*N, 10*N);
    dC   = (A - sum_BR*S) * C;
    dcdt = dC(:);
end

function dedt = eODE(t, e, t_s, s_store, t_p, p_store, A, sum_BR, N)
    S    = lookupMatrix(t, t_s, s_store);
    p    = interp1(t_p, p_store, t, 'linear', 'extrap')';
    dedt = (A - sum_BR*S)*e - sum_BR*p;
end

%% -----------------------------------------------------------------------
%  Control
%% -----------------------------------------------------------------------

function u = computeControl(x, S, p, B, Ri, alpha, N, m, g)
    sp = S*x + p;
    u  = zeros(4*N, 1);
    for i = 1:N
        Bi = B(:, (i-1)*4+1 : i*4);
        u((i-1)*4+1:i*4) = -(1/alpha(i)) * (Ri{i}\Bi') * sp;
    end
    u = inclinationProtection(u, N, m, g);
end

%% -----------------------------------------------------------------------
%  Lookup — nearest-neighbour index into a 3D matrix store
%% -----------------------------------------------------------------------

function M = lookupMatrix(t, t_arr, store)
    idx = interp1(t_arr, 1:length(t_arr), t, 'linear', 'extrap');
    idx = max(1, min(size(store,3), round(idx)));
    M   = store(:,:,idx);
end

%% -----------------------------------------------------------------------
%  Utility
%% -----------------------------------------------------------------------

function u = inclinationProtection(u, N, m, g)
    for i = 1:N
        idx       = (i-1)*4 + 1 : i*4;
        u(idx(1)) = max(-pi/4, min(pi/4,  u(idx(1))));  % pitch
        u(idx(2)) = max(-pi/4, min(pi/4,  u(idx(2))));  % roll
        u(idx(3)) = max(-m*g,  min(2*m*g, u(idx(3))));  % thrust
        u(idx(4)) = max(-2*pi, min(2*pi,  u(idx(4))));  % yaw rate
    end
end

function [t_c_loc, c_store_loc, t_e_loc, e_vec_loc] = solveLocalODEs( ...
    t_start, t_end, t_s, s_store, t_p, p_store, A, sum_BR, N)
tspan = [t_start, t_end];
c0 = eye(10*N); c0 = c0(:);
[t_c_abs, c_vec] = ode45(@(t,cv) cODE(t, cv, t_s, s_store, A, sum_BR, N), tspan, c0);
t_c_loc     = t_c_abs - t_c_abs(1);
c_store_loc = zeros(10*N, 10*N, length(t_c_abs));
for j = 1:length(t_c_abs)
    c_store_loc(:,:,j) = reshape(c_vec(j,:), 10*N, 10*N);
end
e0 = zeros(10*N, 1);
[t_e_abs, e_vec_loc] = ode45(@(t,e) eODE(t, e, t_s, s_store, t_p, p_store, A, sum_BR, N), tspan, e0);
t_e_loc = t_e_abs - t_e_abs(1);
end
