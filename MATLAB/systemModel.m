%% Parameters
g       = 9.81;
m       = 0.027;
a_theta = 1;
a_phi   = 1;
a_r     = 1;
N       = 5;
leader  = 1;
q       = 5;
alpha   = [0.2, 0.2, 0.2, 0.2, 0.2];
t0      = 0;
r_min   = 0.15;
v_mean  = 0.2;
dt      = 0.01;

waypoints = [0,   0,   1;
             1,   0,   1;
             1,   1,   2;
             2,   1,   1;
             2,   0,   1;
           3.5, 0.5,   1;
           3.5, 1.5, 3.5;
           2.5, 2.5, 3.5;
             3, 3.5, 2.5;
             2,   4, 1.5];

% Table I - formation offsets [x, y, z] per drone
formation_offsets = [ 1,  4,  3;
                     -1, -2,  1;
                      3,  1, -2;
                     -3, -3, -1;
                     -4, -4, -4];

% Initial conditions - start at waypoint 1 + formation offsets
x0 = zeros(10*N, 1);
for i = 1:N
    x0((i-1)*10 + 1) = waypoints(1,1) + formation_offsets(i,1);
    x0((i-1)*10 + 4) = waypoints(1,2) + formation_offsets(i,2);
    x0((i-1)*10 + 7) = waypoints(1,3) + formation_offsets(i,3);
end

%% System Matrices
Ai = [0,  1,        0,   0,  0,        0,   0,  0,  0,      0;
      0,  0,        g,   0,  0,        0,   0,  0,  0,      0;
      0,  0,  -a_theta,  0,  0,        0,   0,  0,  0,      0;
      0,  0,        0,   0,  1,        0,   0,  0,  0,      0;
      0,  0,        0,   0,  0,       -g,   0,  0,  0,      0;
      0,  0,        0,   0,  0,   -a_phi,   0,  0,  0,      0;
      0,  0,        0,   0,  0,        0,   0,  1,  0,      0;
      0,  0,        0,   0,  0,        0,   0,  0,  0,      0;
      0,  0,        0,   0,  0,        0,   0,  0,  0,      1;
      0,  0,        0,   0,  0,        0,   0,  0,  0,  -a_r];

bi = [0,        0,    0,     0;
      0,        0,    0,     0;
      a_theta,  0,    0,     0;
      0,        0,    0,     0;
      0,        0,    0,     0;
      0,     a_phi,   0,     0;
      0,        0,    0,     0;
      0,        0,  1/m,     0;
      0,        0,    0,     0;
      0,        0,    0,  a_r];

E_N = eye(N);
E_n = eye(10);
A   = kron(E_N, Ai);
B   = kron(E_N, bi);

%% Graph Matrices
D = [  1,   1,  0,  0;
      -1,   0,  1,  0;
      -1,   0,  0,  1;
       0,  -1,  0,  0;
       0,   0, -1,  0];

D_hat = kron(D, E_n);

W = [5, 5, 0, 0;
     5, 0, 5, 0;
     5, 0, 0, 5;
     0, 5, 0, 0;
     0, 0, 5, 0];

%% Q and R Matrices
W_hat = cell(N, 1);
Q     = cell(N, 1);
Ri    = cell(N, 1);

for i = 1:N
    W_hat{i} = kron(diag(W(i,:)), E_n);
    Q{i}     = D_hat * W_hat{i} * D_hat';
    Ri{i}    = eye(4);
end

% Leader trajectory tracking term
q_diag                                       = zeros(10*N, 10*N);
row_start                                    = (leader-1)*10 + 1;
row_end                                      = leader*10;
q_diag(row_start:row_end, row_start:row_end) = q * E_n;
Q{leader}                                    = Q{leader} + q_diag;

% Precompute sum of alpha_i * Qi
sum_alpha_Q = zeros(10*N, 10*N);
for i = 1:N
    sum_alpha_Q = sum_alpha_Q + alpha(i) * Q{i};
end

% Precompute sum_BR (time invariant)
sum_BR = zeros(10*N, 10*N);
for i = 1:N
    Bi_block = B(:, (i-1)*4+1 : i*4);
    sum_BR   = sum_BR + (1/alpha(i)) * Bi_block * (Ri{i} \ eye(4)) * Bi_block';
end

%% Path Manager
[xd, tT] = pathManager(waypoints, formation_offsets, v_mean, dt, r_min, g);
n_steps   = size(xd, 1);

fprintf('Mission time tT = %.2f s\n', tT);
fprintf('n_steps = %d\n', n_steps);

%% Riccati Solver
tspan = [tT, t0];

% --- s(t) ---
sT     = sum_alpha_Q;
sT_vec = sT(:);

[t_s, s_vec] = ode45(@(t, s_vec) riccatiODE(t, s_vec, A, B, Ri, alpha, sum_alpha_Q, N), tspan, sT_vec);

s_store = zeros(10*N, 10*N, length(t_s));
for k = 1:length(t_s)
    s_store(:,:,k) = reshape(s_vec(k,:), 10*N, 10*N);
end

t_s     = flip(t_s);
s_store = flip(s_store, 3);

% --- p(t) ---
pT     = -sum_alpha_Q * xd(end,:)';
pT_vec = pT;

[t_p, p_vec] = ode45(@(t, p_vec) pODE(t, p_vec, t_s, s_store, A, sum_alpha_Q, sum_BR, xd, dt, N), tspan, pT_vec);

t_p     = flip(t_p);
p_store = flip(p_vec, 1);

%% Simulation (Closed Loop)
x_store = zeros(10*N, n_steps);
x_store(:,1) = x0;
u_store = zeros(4*N, n_steps);

for k = 1:n_steps-1
    t_k = (k-1) * dt;

    x_k = x_store(:, k);

    % Interpolate s and p at current time
    s_k = interpS(t_k, t_s, s_store, N);
    p_k = interp1(t_p, p_store, t_k, 'linear', 'extrap')';

    % Compute control input for each drone
    u_k = zeros(4*N, 1);
    for i = 1:N
        Bi_block            = B(:, (i-1)*4+1 : i*4);
        u_k((i-1)*4+1:i*4) = -(1/alpha(i)) * (Ri{i} \ Bi_block') * (s_k * x_k + p_k);
    end

    % Apply inclination protection
    u_k = inclinationProtection(u_k, N, m, g);

    u_store(:, k)      = u_k;
    x_store(:, k+1)    = x_k + dt * (A * x_k + B * u_k);
end

%% Performance (TATD)
skip    = round(5 / dt);
x_eval  = x_store(:, skip:end);
xd_eval = xd(skip:end, :)';
NT      = size(x_eval, 2);

TATD = 0;
for k = 1:NT
    e    = x_eval(:,k) - xd_eval(:,k);
    TATD = TATD + (e' * e);
end
TATD = sqrt(TATD / (N * NT));
fprintf('TATD = %.6f meters\n', TATD);

%% Plots
t_vec = (0:n_steps-1) * dt;

% Central path vs waypoints
figure;
% Central path is drone 1 x,y minus its formation offset
x_central = xd(:,1) - formation_offsets(1,1);
y_central = xd(:,4) - formation_offsets(1,2);
plot(x_central, y_central, 'w-', 'LineWidth', 1.5); hold on;
plot(waypoints(:,1), waypoints(:,2), 'r*', 'MarkerSize', 10);
title('Central path vs waypoints');
xlabel('x (m)'); ylabel('y (m)'); grid on;
legend('Central path', 'Waypoints');

% Agent trajectories
figure; hold on;
colors = lines(N);
for i = 1:N
    x_idx = (i-1)*10 + 1;
    y_idx = (i-1)*10 + 4;
    plot(x_store(x_idx,:), x_store(y_idx,:), 'Color', colors(i,:));
    plot(x_store(x_idx,1), x_store(y_idx,1), 'o', ...
         'MarkerSize', 8, 'MarkerFaceColor', colors(i,:), 'Color', colors(i,:));
end
plot(waypoints(:,1), waypoints(:,2), 'k--', 'LineWidth', 1.5);
plot(waypoints(:,1), waypoints(:,2), 'k*', 'MarkerSize', 10);
title('Agent x-y positions');
xlabel('x (m)'); ylabel('y (m)'); grid on;
legend('Drone 1','Drone 2','Drone 3','Drone 4','Drone 5', ...
       'Init 1','Init 2','Init 3','Init 4','Init 5','Waypoints');

% s(1,1) diagnostic
figure;
plot(t_s, squeeze(s_store(1,1,:)));
title('s(1,1) over time');
xlabel('t (s)'); grid on;

% p(1) diagnostic
figure;
plot(t_p, p_store(:,1));
title('p(1) over time');
xlabel('t (s)'); grid on;

% Control inputs drone 1
figure;
subplot(4,1,1); plot(t_vec, u_store(1,:)); ylabel('\theta_d'); title('Drone 1 control inputs'); grid on;
subplot(4,1,2); plot(t_vec, u_store(2,:)); ylabel('\phi_d'); grid on;
subplot(4,1,3); plot(t_vec, u_store(3,:)); ylabel('\deltaF'); grid on;
subplot(4,1,4); plot(t_vec, u_store(4,:)); ylabel('r_d'); xlabel('t (s)'); grid on;

saveas(gcf, '/home/jonah/Downloads/plot.png')


% Plot central path (remove drone 1 offset to get back to central)
figure;
plot(xd(:,1) - formation_offsets(1,1), xd(:,4) - formation_offsets(1,2), 'w-');
hold on;
plot(waypoints(:,1), waypoints(:,2), 'r*', 'MarkerSize', 10);
title('Central path check');
%% Functions
function [xd, tT] = pathManager(waypoints, formation_offsets, v_mean, dt, r_min, g)
    path = waypoints(1,:);

    for i = 2:size(waypoints,1)-1
        A_wp = waypoints(i-1,:);
        B_wp = waypoints(i,  :);
        C_wp = waypoints(i+1,:);

        AB = B_wp - A_wp;
        BC = C_wp - B_wp;

        unit_AB = AB / norm(AB);
        unit_BC = BC / norm(BC);

        cos_sigma   = max(-1, min(1, dot(unit_AB, unit_BC)));
        sigma       = acos(cos_sigma);
        delta_sigma = pi - sigma;
        r_real      = r_min * tan(delta_sigma / 2);

        l = B_wp - r_real * unit_AB;
        d = B_wp + r_real * unit_BC;

        n_vec = cross(unit_AB, unit_BC);
        if norm(n_vec) < 1e-10
            path = [path; B_wp];
            continue;
        end
        n_vec = n_vec / norm(n_vec);

        perp_AB = cross(n_vec, unit_AB);
        perp_AB = perp_AB / norm(perp_AB);

        R      = r_real * tan(delta_sigma / 2);
        center = l + R * perp_AB;

        n_arc          = max(2, round(delta_sigma * R / (v_mean * dt)));
        angle_per_step = delta_sigma / n_arc;
        e1             = (l - center) / norm(l - center);

        path = [path; l];
        for k = 1:n_arc
            theta = k * angle_per_step;
            point = center + R * rodrigues(e1, n_vec, theta);
            path  = [path; point];
        end
    end

    path = [path; waypoints(end,:)];

    % Uniform arc length resampling for constant speed
    dists     = cumsum([0, sqrt(sum(diff(path).^2, 2))']);
    tT        = dists(end) / v_mean;
    s_uniform = 0:v_mean*dt:dists(end);
    pos       = interp1(dists, path, s_uniform, 'linear', 'extrap');
    T         = size(pos, 1);

    % Velocity and acceleration by finite difference
    vel            = diff(pos) / dt;
    vel(end+1,:)   = vel(end,:);
    accel          = diff(vel) / dt;
    accel(end+1,:) = accel(end,:);

    % Desired attitudes from linearised dynamics
    theta_d = accel(:,1) / g;
    phi_d   = -accel(:,2) / g;

    % Central path desired state
    xd_central       = zeros(T, 10);
    xd_central(:,1)  = pos(:,1);
    xd_central(:,2)  = vel(:,1);
    xd_central(:,3)  = theta_d;
    xd_central(:,4)  = pos(:,2);
    xd_central(:,5)  = vel(:,2);
    xd_central(:,6)  = phi_d;
    xd_central(:,7)  = pos(:,3);
    xd_central(:,8)  = vel(:,3);
    xd_central(:,9)  = 0;
    xd_central(:,10) = 0;

    % Expand to all drones with formation offsets
    N_drones = size(formation_offsets, 1);
    xd       = zeros(T, 10*N_drones);
    for i = 1:N_drones
        idx           = (i-1)*10 + 1 : i*10;
        xd(:, idx)    = xd_central;
        xd(:, idx(1)) = xd_central(:,1) + formation_offsets(i,1);
        xd(:, idx(4)) = xd_central(:,4) + formation_offsets(i,2);
        xd(:, idx(7)) = xd_central(:,7) + formation_offsets(i,3);
    end
end

function u = inclinationProtection(u, N, m, g)
    for i = 1:N
        idx       = (i-1)*4 + 1 : i*4;
        u(idx(1)) = max(-pi/4,  min(pi/4,    u(idx(1))));  % theta_d
        u(idx(2)) = max(-pi/4,  min(pi/4,    u(idx(2))));  % phi_d
        u(idx(3)) = max(0,      min(2*m*g,   u(idx(3))));  % thrust
        u(idx(4)) = max(-2*pi,  min(2*pi,    u(idx(4))));  % yaw rate
    end
end

function v_rot = rodrigues(v, axis, theta)
    v_rot = v*cos(theta) + cross(axis,v)*sin(theta) + axis*dot(axis,v)*(1-cos(theta));
end

function dsdt = riccatiODE(t, s_vec, A, B, Ri, alpha, sum_alpha_Q, N)
    s      = reshape(s_vec, 10*N, 10*N);
    sum_BR = zeros(10*N, 10*N);
    for i = 1:N
        Bi_block = B(:, (i-1)*4+1 : i*4);
        sum_BR   = sum_BR + (1/alpha(i)) * Bi_block * (Ri{i} \ eye(4)) * Bi_block';
    end
    % Equation 9b
    dsdt = -s*A - A'*s - sum_alpha_Q + s * sum_BR * s;
    dsdt = dsdt(:);
end

function dpdt = pODE(t, p, t_s, s_store, A, sum_alpha_Q, sum_BR, xd, dt, N)
    % Interpolate s at current time
    s = interpS(t, t_s, s_store, N);

    % Interpolate xd at current time
    k_xd = max(1, min(size(xd,1), round(t / dt) + 1));
    xd_t = xd(k_xd,:)';

    % Equation 9c — from Aghajani Eq 11
    dpdt = (s * sum_BR - A') * p + sum_alpha_Q * xd_t;
end

function s = interpS(t, t_s, s_store, N)
    idx = interp1(t_s, 1:length(t_s), t, 'linear', 'extrap');
    idx = max(1, min(length(t_s), round(idx)));
    s   = s_store(:,:,idx);
end