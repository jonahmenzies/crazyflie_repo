function [x_cl, u_cl, TATD_cl] = closedLoop(sys, ric)
%CLOSEDLOOP  Closed-loop formation control simulation.
%   Full state feedback at every timestep:
%     u(t) = computeControl(x(t), S(t), p(t))
%
%   Inputs:
%     sys  -- params struct (P): A, B, Ri, alpha, x0, N, m, g, dt
%     ric  -- riccati struct: t_s, s_store, t_p, p_store, xd, n_steps
%
%   Outputs:
%     x_cl    -- state trajectory   (10N x n_steps)
%     u_cl    -- control trajectory  (4N x n_steps)
%     TATD_cl -- time-averaged tracking deviation (m)

A  = sys.A;   B  = sys.B;   Ri    = sys.Ri;
N  = sys.N;   m  = sys.m;   g     = sys.g;
dt = sys.dt;  x0 = sys.x0;  alpha = sys.alpha;

t_s = ric.t_s;  s_store = ric.s_store;
t_p = ric.t_p;  p_store = ric.p_store;
xd  = ric.xd;   n_steps = ric.n_steps;

x_cl    = zeros(10*N, n_steps);
x_cl(:,1) = x0;
u_cl    = zeros(4*N, n_steps);

for k = 1:n_steps-1
    t_k = (k-1) * dt;
    s_k = utils.lookupMatrix(t_k, t_s, s_store);
    p_k = interp1(t_p, p_store, t_k, 'linear', 'extrap')';
    u_k = utils.computeControl(x_cl(:,k), s_k, p_k, B, Ri, alpha, N, m, g);

    u_cl(:,k)   = u_k;
    x_cl(:,k+1) = x_cl(:,k) + dt * (A*x_cl(:,k) + B*u_k);
end

TATD_cl = utils.computeTATD(x_cl, xd, N, n_steps, dt);
end