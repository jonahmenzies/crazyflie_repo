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
load('riccati_solution.mat', 't_s', 's_store', 't_p', 'p_store', 'xd', 'n_steps');

%% Closed-Loop Simulation
fprintf('Running closed-loop simulation...\n');
x_store = zeros(10*N, n_steps);
x_store(:,1) = x0;
u_store = zeros(4*N, n_steps);

for k = 1:n_steps-1
    t_k = (k-1)*dt;
    s_k = lookupMatrix(t_k, t_s, s_store);
    p_k = interp1(t_p, p_store, t_k, 'linear', 'extrap')';
    u_k = computeControl(x_store(:,k), s_k, p_k, B, Ri, alpha, N, m, g);

    u_store(:,k)   = u_k;
    x_store(:,k+1) = x_store(:,k) + dt*(A*x_store(:,k) + B*u_k);
end

%% Performance (TATD)
skip = round(5/dt);
NT   = n_steps - skip;

pos_idx = [];
for i = 1:N
    base    = (i-1)*10;
    pos_idx = [pos_idx, base+1, base+4, base+7];
end

err_cl  = x_store(pos_idx, skip:end) - xd(skip:end, pos_idx)';
TATD_cl = sqrt(sum(err_cl(:).^2) / (N*NT));

fprintf('Closed-Loop TATD = %.6f metres\n', TATD_cl);

%% Plots
t_vec      = (0:n_steps-1)*dt;
colors     = lines(N);
drone_lbls = {'Drone 1','Drone 2','Drone 3','Drone 4','Drone 5'};

figure;
h1 = plot(t_vec, xd(:,1),      'b--', 'LineWidth', 1.5); hold on;
h2 = plot(t_vec, x_store(1,:), 'r-',  'LineWidth', 1);
title('Drone 1 — Demanded vs Actual x');
xlabel('Time (s)'); ylabel('x (m)');
legend([h1,h2], 'Demanded','Closed Loop'); grid on;

figure;
subplot(2,1,1); hold on;
h_cl = gobjects(N,1);
for i = 1:N
    h_cl(i) = plot(t_vec, x_store((i-1)*10+1,:), 'Color', colors(i,:));
end
title('Closed Loop x — All Drones');
xlabel('Time (s)'); ylabel('x (m)');
legend(h_cl, drone_lbls); grid on;

subplot(2,1,2); hold on;
h_u = gobjects(N,1);
for i = 1:N
    h_u(i) = plot(t_vec, u_store((i-1)*4+3,:), 'Color', colors(i,:));
end
title('Closed Loop Thrust — All Drones');
xlabel('Time (s)'); ylabel('Thrust (N)');
legend(h_u, drone_lbls); grid on;

figure; hold on;
h_cl2 = gobjects(N,1);
for i = 1:N
    h_cl2(i) = plot(x_store((i-1)*10+1,:), x_store((i-1)*10+4,:), 'Color', colors(i,:));
    plot(x_store((i-1)*10+1,1), x_store((i-1)*10+4,1), 'o', ...
         'MarkerSize', 8, 'MarkerFaceColor', colors(i,:), 'Color', colors(i,:));
end
title('Closed Loop — x vs y');
xlabel('x (m)'); ylabel('y (m)'); grid on; axis equal;
legend(h_cl2, drone_lbls);

figure;
h3d_cl = plot3(x_store(1,:), x_store(4,:), x_store(7,:), 'r-',  'LineWidth', 1.5); hold on;
h3d_d  = plot3(xd(:,1),      xd(:,4),      xd(:,7),      'b--', 'LineWidth', 1.5);
h3d_s  = plot3(x_store(1,1), x_store(4,1), x_store(7,1), 'ko',  ...
               'MarkerSize', 10, 'MarkerFaceColor', 'k');
title('Drone 1 — 3D Trajectory');
xlabel('x (m)'); ylabel('y (m)'); zlabel('z (m)');
legend([h3d_cl,h3d_d,h3d_s], 'Closed Loop','Demanded','Start');
grid on; view(45, 30);

saveas(gcf, '/home/jonah/Downloads/plot_cl.png');

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