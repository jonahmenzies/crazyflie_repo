function riccatiSolver(P)
%RICCATISOLVER  Solve cooperative Riccati equations and save solution.
%   P must be supplied by the caller (e.g. sims.params(), possibly with
%   overridden x0/waypoints) -- no local redefinition.
%   Saves riccati_solution.mat for use by simulation scripts.
%
%   Usage:
%       P = sims.params();
%       sims.riccatiSolver(P);

% Delete previous solution file if it exists
if exist('riccati_solution.mat', 'file') == 2
    delete('riccati_solution.mat');
    fprintf('  Deleted existing riccati_solution.mat\n');
end

N      = P.N;
alpha  = P.alpha;
leader = P.leader;
q      = P.q;
dt     = P.dt;
t0     = P.t0;
A      = P.A;
B      = P.B;
Ri     = P.Ri;
E_n    = P.E_n;
D      = P.D;
W      = P.W;

%-- Graph / Cost Matrices -----------------------------------------
D_hat = kron(D, E_n);
W_hat = cell(N, 1);
Q     = cell(N, 1);
for i = 1:N
    W_hat{i} = kron(diag(W(i,:)), E_n);
    Q{i}     = D_hat * W_hat{i} * D_hat';
end

% Leader tracking term
q_diag = zeros(10*N, 10*N);
row_start = (leader-1)*10 + 1;
row_end   = leader*10;
q_diag(row_start:row_end, row_start:row_end) = q * E_n;
Q{leader} = Q{leader} + q_diag;

% Team-weighted aggregates
sum_alpha_Q = zeros(10*N, 10*N);
sum_BR      = zeros(10*N, 10*N);
for i = 1:N
    Bi          = B(:, (i-1)*4+1 : i*4);
    sum_alpha_Q = sum_alpha_Q + alpha(i) * Q{i};
    sum_BR      = sum_BR + (1/alpha(i)) * Bi * (Ri{i} \ Bi');
end

%-- Reference Trajectory ------------------------------------------
[xd, tT] = utils.pathManager(P.waypoints, P.formation_offsets, ...
                              P.v_mean, dt, P.r_min, P.g);
n_steps  = size(xd, 1);
fprintf('  Mission time tT = %.2f s,  n_steps = %d\n', tT, n_steps);

%-- S(t): backward integration from tT to t0 ----------------------
tspan = [tT, t0];
sT = sum_alpha_Q;
sT = sT(:);
[t_s, s_vec] = ode45( ...
    @(t,sv) utils.odes.sODE(t, sv, A, sum_alpha_Q, sum_BR, N), tspan, sT);
s_store = zeros(10*N, 10*N, length(t_s));
for k = 1:length(t_s)
    s_store(:,:,k) = reshape(s_vec(k,:), 10*N, 10*N);
end
t_s     = flip(t_s);
s_store = flip(s_store, 3);

%-- p(t): backward integration from tT to t0 ----------------------
pT = -sum_alpha_Q * xd(end,:)';
[t_p, p_vec] = ode45( ...
    @(t,p) utils.odes.pODE(t, p, t_s, s_store, A, sum_alpha_Q, sum_BR, xd, dt, N), ...
    tspan, pT);
t_p     = flip(t_p);
p_store = flip(p_vec, 1);

%-- Save ----------------------------------------------------------
save('riccati_solution.mat', ...
    't_s', 's_store', 't_p', 'p_store', 'sum_BR', 'xd', 'tT', 'n_steps');
fprintf('  Saved riccati_solution.mat\n');
end