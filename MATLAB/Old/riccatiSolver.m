%%
clc; clear all; close all;

%% Parameters
params;

%% Multi-agent system matrices
E_N = eye(N);
E_n = eye(10);
A   = kron(E_N, Ai);
B   = kron(E_N, bi);

%% Graph Matrices
D = [ 1,  1,  0,  0;
     -1,  0,  1,  0;
     -1,  0,  0,  1;
      0, -1,  0,  0;
      0,  0, -1,  0];

D_hat = kron(D, E_n);

W = [5, 5, 0, 0;
     5, 0, 5, 0;
     5, 0, 0, 5;
     0, 5, 0, 0;
     0, 0, 5, 0];

%% Cost Matrices
W_hat  = cell(N, 1);
Q      = cell(N, 1);
Ri     = cell(N, 1);
R_vals = [5600, 1005, 1050, 1050, 2920];

for i = 1:N
    W_hat{i} = kron(diag(W(i,:)), E_n);
    Q{i}     = D_hat * W_hat{i} * D_hat';
    Ri{i}    = R_vals(i) * eye(4);
end

% Leader trajectory tracking term
q_diag                                       = zeros(10*N, 10*N);
row_start                                    = (leader-1)*10 + 1;
row_end                                      = leader*10;
q_diag(row_start:row_end, row_start:row_end) = q * E_n;
Q{leader}                                    = Q{leader} + q_diag;

% Team-weighted aggregates — precomputed once, time-invariant
sum_alpha_Q = zeros(10*N, 10*N);
sum_BR      = zeros(10*N, 10*N);
for i = 1:N
    Bi_block    = B(:, (i-1)*4+1 : i*4);
    sum_alpha_Q = sum_alpha_Q + alpha(i) * Q{i};
    sum_BR      = sum_BR + (1/alpha(i)) * Bi_block * (Ri{i} \ Bi_block');
end

%% Path Manager
[xd, tT] = pathManager(waypoints, formation_offsets, v_mean, dt, r_min, g);
n_steps   = size(xd, 1);

fprintf('Mission time tT = %.2f s\n', tT);
fprintf('n_steps = %d\n', n_steps);

%% Riccati Solver (backward in time)
tspan = [tT, t0];

% S(t) — cooperative Riccati equation, terminal condition S(T) = sum_alpha_Q
sT    = sum_alpha_Q;
sT    = sT(:);

[t_s, s_vec] = ode45(@(t, sv) riccatiODE(t, sv, A, sum_alpha_Q, sum_BR, N), tspan, sT);

s_store = zeros(10*N, 10*N, length(t_s));
for k = 1:length(t_s)
    s_store(:,:,k) = reshape(s_vec(k,:), 10*N, 10*N);
end
t_s     = flip(t_s);
s_store = flip(s_store, 3);

% p(t) — feedforward tracking term, terminal condition p(T) = -sum_alpha_Q * xd(T)
% Note: corrects missing -A'p term from Jiang et al. (2020) eq. 9c
pT    = -sum_alpha_Q * xd(end,:)';
[t_p, p_vec] = ode45(@(t, p) pODE(t, p, t_s, s_store, A, sum_alpha_Q, sum_BR, xd, dt, N), tspan, pT);

t_p     = flip(t_p);
p_store = flip(p_vec, 1);

%% Save
save('riccati_solution.mat', 't_s', 's_store', 't_p', 'p_store', 'sum_BR', 'xd', 'tT', 'n_steps');
fprintf('Saved riccati_solution.mat\n');

%% -----------------------------------------------------------------------
%  ODE Right-Hand Sides
%% -----------------------------------------------------------------------

function dsdt = riccatiODE(~, sv, A, sum_alpha_Q, sum_BR, N)
    S    = reshape(sv, 10*N, 10*N);
    dS   = -S*A - A'*S - sum_alpha_Q + S*sum_BR*S;
    dsdt = dS(:);
end

function dpdt = pODE(t, p, t_s, s_store, A, sum_alpha_Q, sum_BR, xd, dt, N)
    S    = lookupMatrix(t, t_s, s_store);
    k    = max(1, min(size(xd,1), round(t/dt) + 1));
    dpdt = (S*sum_BR - A')*p + sum_alpha_Q*xd(k,:)';
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

        n_vec = cross(unit_AB, unit_BC);
        if norm(n_vec) < 1e-10
            path = [path; B_wp];
            continue;
        end
        n_vec = n_vec / norm(n_vec);

        perp_AB = cross(n_vec, unit_AB);
        perp_AB = perp_AB / norm(perp_AB);

        R      = r_real * tan(delta_sigma / 2);
        l      = B_wp - r_real * unit_AB;
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

    dists     = cumsum([0, sqrt(sum(diff(path).^2, 2))']);
    tT        = dists(end) / v_mean;
    s_uniform = 0:v_mean*dt:dists(end);
    pos       = interp1(dists, path, s_uniform, 'linear', 'extrap');
    T         = size(pos, 1);

    vel            = diff(pos) / dt;
    vel(end+1,:)   = vel(end,:);
    accel          = diff(vel) / dt;
    accel(end+1,:) = accel(end,:);

    theta_d = accel(:,1) / g;
    phi_d   = -accel(:,2) / g;

    xd_central        = zeros(T, 10);
    xd_central(:,1)   = pos(:,1);
    xd_central(:,2)   = vel(:,1);
    xd_central(:,3)   = theta_d;
    xd_central(:,4)   = pos(:,2);
    xd_central(:,5)   = vel(:,2);
    xd_central(:,6)   = phi_d;
    xd_central(:,7)   = pos(:,3);
    xd_central(:,8)   = vel(:,3);
    xd_central(:,9)   = 0;
    xd_central(:,10)  = 0;

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

function v_rot = rodrigues(v, axis, theta)
    v_rot = v*cos(theta) + cross(axis,v)*sin(theta) + axis*dot(axis,v)*(1-cos(theta));
end