%%
clc; clear all; close all;

%% Parameters
params;

%% Multi-agent system matrices
A = kron(eye(N), Ai);
B = kron(eye(N), bi);

%% Cost Matrices (R only — S and p loaded from riccatiSolver)
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

% c(t) — state transition matrix of closed-loop system, initial condition c(0) = I
c0_vec = eye(10*N);
c0_vec = c0_vec(:);
[t_c, c_vec_ode] = ode45(@(t, cv) cODE(t, cv, t_s, s_store, A, sum_BR, N), tspan_fwd, c0_vec);

c_store = zeros(10*N, 10*N, length(t_c));
for k = 1:length(t_c)
    c_store(:,:,k) = reshape(c_vec_ode(k,:), 10*N, 10*N);
end

% e(t) — forced response due to reference trajectory, initial condition e(0) = 0
e0 = zeros(10*N, 1);
[t_e, e_vec_ode] = ode45(@(t, e) eODE(t, e, t_s, s_store, t_p, p_store, A, sum_BR, N), tspan_fwd, e0);

%% Open-Loop Simulation
fprintf('Running open-loop simulation...\n');
x_ol = zeros(10*N, n_steps);
x_ol(:,1) = x0;
u_ol = zeros(4*N, n_steps);

for k = 1:n_steps-1
    t_k   = (k-1)*dt;
    s_k   = lookupMatrix(t_k, t_s, s_store);
    p_k   = interp1(t_p, p_store, t_k, 'linear', 'extrap')';
    c_k   = lookupMatrix(t_k, t_c, c_store);
    e_k   = interp1(t_e, e_vec_ode, t_k, 'linear', 'extrap')';
    x_hat = c_k*x0 + e_k;
    u_k   = computeControl(x_hat, s_k, p_k, B, Ri, alpha, N, m, g);

    u_ol(:,k)   = u_k;
    x_ol(:,k+1) = x_ol(:,k) + dt*(A*x_ol(:,k) + B*u_k);
end

%% Performance (TATD)
skip = round(5/dt);
NT   = n_steps - skip;

pos_idx = [];
for i = 1:N
    base    = (i-1)*10;
    pos_idx = [pos_idx, base+1, base+4, base+7];
end

err_ol  = x_ol(pos_idx, skip:end) - xd(skip:end, pos_idx)';
TATD_ol = sqrt(sum(err_ol(:).^2) / (N*NT));

fprintf('Open-Loop TATD = %.6f metres\n', TATD_ol);

%% Plots
t_vec      = (0:n_steps-1)*dt;
colors     = lines(N);
drone_lbls = {'Drone 1','Drone 2','Drone 3','Drone 4','Drone 5'};

figure;
h1 = plot(t_vec, xd(:,1),   'b--', 'LineWidth', 1.5); hold on;
h2 = plot(t_vec, x_ol(1,:), 'g-',  'LineWidth', 1);
title('Drone 1 — Demanded vs Actual x');
xlabel('Time (s)'); ylabel('x (m)');
legend([h1,h2], 'Demanded','Open Loop'); grid on;

figure;
subplot(2,1,1); hold on;
h_ol = gobjects(N,1);
for i = 1:N
    h_ol(i) = plot(t_vec, x_ol((i-1)*10+1,:), 'Color', colors(i,:));
end
title('Open Loop x — All Drones');
xlabel('Time (s)'); ylabel('x (m)');
legend(h_ol, drone_lbls); grid on;

subplot(2,1,2); hold on;
h_u = gobjects(N,1);
for i = 1:N
    h_u(i) = plot(t_vec, u_ol((i-1)*4+3,:), 'Color', colors(i,:));
end
title('Open Loop Thrust — All Drones');
xlabel('Time (s)'); ylabel('Thrust (N)');
legend(h_u, drone_lbls); grid on;

figure; hold on;
h_ol2 = gobjects(N,1);
for i = 1:N
    h_ol2(i) = plot(x_ol((i-1)*10+1,:), x_ol((i-1)*10+4,:), 'Color', colors(i,:));
    plot(x_ol((i-1)*10+1,1), x_ol((i-1)*10+4,1), 'o', ...
         'MarkerSize', 8, 'MarkerFaceColor', colors(i,:), 'Color', colors(i,:));
end
title('Open Loop — x vs y');
xlabel('x (m)'); ylabel('y (m)'); grid on; axis equal;
legend(h_ol2, drone_lbls);

figure;
h3d_ol = plot3(x_ol(1,:), x_ol(4,:), x_ol(7,:), 'g-',  'LineWidth', 1.5); hold on;
h3d_d  = plot3(xd(:,1),   xd(:,4),   xd(:,7),   'b--', 'LineWidth', 1.5);
h3d_s  = plot3(x_ol(1,1), x_ol(4,1), x_ol(7,1), 'ko',  ...
               'MarkerSize', 10, 'MarkerFaceColor', 'k');
title('Drone 1 — 3D Trajectory');
xlabel('x (m)'); ylabel('y (m)'); zlabel('z (m)');
legend([h3d_ol,h3d_d,h3d_s], 'Open Loop','Demanded','Start');
grid on; view(45, 30);

saveas(gcf, '/home/jonah/Downloads/plot_ol.png');

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