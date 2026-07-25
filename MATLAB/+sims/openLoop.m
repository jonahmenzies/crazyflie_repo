function [x_ol, u_ol, TATD_ol] = openLoop(sys, ric)
%OPENLOOP  Open-loop formation control simulation.
%   Control computed from predicted state x_hat, never from actual state:
%     x_hat(t) = c(t)*x0 + e(t)
%     u(t)     = computeControl(x_hat, S(t), p(t))
%
%   Inputs:
%     sys  -- params struct (P): A, B, Ri, alpha, x0, N, m, g, dt, t0
%     ric  -- riccati struct: t_s, s_store, t_p, p_store, sum_BR, xd, tT, n_steps
%
%   Outputs:
%     x_ol    -- state trajectory   (10N x n_steps)
%     u_ol    -- control trajectory  (4N x n_steps)
%     TATD_ol -- time-averaged tracking deviation (m)

    A  = sys.A;   B  = sys.B;   Ri    = sys.Ri;
    N  = sys.N;   m  = sys.m;   g     = sys.g;
    dt = sys.dt;  t0 = sys.t0;  x0    = sys.x0;
    alpha = sys.alpha;

    t_s    = ric.t_s;   s_store = ric.s_store;
    t_p    = ric.t_p;   p_store = ric.p_store;
    sum_BR = ric.sum_BR; xd     = ric.xd;
    tT     = ric.tT;    n_steps = ric.n_steps;

    %-- Solve c and e over full mission horizon -----------------------
    tspan = [t0, tT];

    c0 = eye(10*N); c0 = c0(:);
    [t_c, c_vec] = ode45( ...
        @(t,cv) utils.odes.cODE(t, cv, t_s, s_store, A, sum_BR, N), tspan, c0);

    c_store = zeros(10*N, 10*N, length(t_c));
    for j = 1:length(t_c)
        c_store(:,:,j) = reshape(c_vec(j,:), 10*N, 10*N);
    end

    e0 = zeros(10*N, 1);
    [t_e, e_vec] = ode45( ...
        @(t,e) utils.odes.eODE(t, e, t_s, s_store, t_p, p_store, A, sum_BR, N), tspan, e0);

    %-- Simulation ----------------------------------------------------
    x_ol    = zeros(10*N, n_steps);
    x_ol(:,1) = x0;
    u_ol    = zeros(4*N, n_steps);

    for k = 1:n_steps-1
        t_k   = (k-1) * dt;
        s_k   = utils.lookupMatrix(t_k, t_s, s_store);
        p_k   = interp1(t_p, p_store, t_k, 'linear', 'extrap')';
        c_k   = utils.lookupMatrix(t_k, t_c, c_store);
        e_k   = interp1(t_e, e_vec, t_k, 'linear', 'extrap')';
        x_hat = c_k*x0 + e_k;
        u_k   = utils.computeControl(x_hat, s_k, p_k, B, Ri, alpha, N, m, g);

        u_ol(:,k)   = u_k;
        x_ol(:,k+1) = x_ol(:,k) + dt * (A*x_ol(:,k) + B*u_k);
    end

    TATD_ol = utils.computeTATD(x_ol, xd, N, n_steps, dt);
end