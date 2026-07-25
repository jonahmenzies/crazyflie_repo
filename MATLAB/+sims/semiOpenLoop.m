function [x_sol, u_sol, TATD_sol] = semiOpenLoop(sys, ric, replan_interval)
%SEMIOPENLOOP  Semi-open-loop formation control simulation.
%   c and e pre-solved from each replan start to tT. At each ping,
%   x0_plan updates and the correct pre-solved c/e interval is loaded.
%   Missed pings: tau keeps running -- c/e valid to tT so no blowup.
%
%   Inputs:
%     sys              -- params struct (P): A, B, Ri, alpha, x0, N, m, g, dt, t0
%     ric              -- riccati struct: t_s, s_store, t_p, p_store, sum_BR, xd, tT, n_steps
%     replan_interval  -- seconds between pings
%
%   Outputs:
%     x_sol    -- state trajectory   (10N x n_steps)
%     u_sol    -- control trajectory  (4N x n_steps)
%     TATD_sol -- time-averaged tracking deviation (m)

    A  = sys.A;   B  = sys.B;   Ri    = sys.Ri;
    N  = sys.N;   m  = sys.m;   g     = sys.g;
    dt = sys.dt;  t0 = sys.t0;  x0    = sys.x0;
    alpha = sys.alpha;

    t_s    = ric.t_s;   s_store = ric.s_store;
    t_p    = ric.t_p;   p_store = ric.p_store;
    sum_BR = ric.sum_BR; xd     = ric.xd;
    tT     = ric.tT;    n_steps = ric.n_steps;

    replan_steps  = round(replan_interval / dt);
    replan_starts = t0 : replan_interval : tT - dt;
    n_intervals   = length(replan_starts);

    %-- Pre-compute c and e for every replan interval (offline) -------
    t_c_all = cell(n_intervals, 1);
    c_all   = cell(n_intervals, 1);
    t_e_all = cell(n_intervals, 1);
    e_all   = cell(n_intervals, 1);

    for r = 1:n_intervals
        tspan = [replan_starts(r), tT];

        c0_r = eye(10*N); c0_r = c0_r(:);
        [t_c_abs, c_vec] = ode45( ...
            @(t,cv) utils.odes.cODE(t, cv, t_s, s_store, A, sum_BR, N), tspan, c0_r);
        t_c_all{r} = t_c_abs - t_c_abs(1);
        c_all{r}   = zeros(10*N, 10*N, length(t_c_abs));
        for j = 1:length(t_c_abs)
            c_all{r}(:,:,j) = reshape(c_vec(j,:), 10*N, 10*N);
        end

        e0_r = zeros(10*N, 1);
        [t_e_abs, e_vec] = ode45( ...
            @(t,e) utils.odes.eODE(t, e, t_s, s_store, t_p, p_store, A, sum_BR, N), tspan, e0_r);
        t_e_all{r} = t_e_abs - t_e_abs(1);
        e_all{r}   = e_vec;
    end

    %-- Simulation ----------------------------------------------------
    x_sol    = zeros(10*N, n_steps);
    x_sol(:,1) = x0;
    u_sol    = zeros(4*N, n_steps);

    x0_plan = x0;
    tau     = 0;
    r_idx   = 1;

    for k = 1:n_steps-1
        if mod(k-1, replan_steps) == 0 && k > 1
            r_idx   = min(r_idx + 1, n_intervals);
            x0_plan = x_sol(:,k);
            tau     = 0;
        end

        t_k   = (k-1) * dt;
        t_tau = tau * dt;

        s_k   = utils.lookupMatrix(t_k,   t_s,           s_store);
        p_k   = interp1(t_p, p_store, t_k, 'linear', 'extrap')';
        c_k   = utils.lookupMatrix(t_tau, t_c_all{r_idx}, c_all{r_idx});
        e_k   = interp1(t_e_all{r_idx}, e_all{r_idx}, t_tau, 'linear', 'extrap')';

        x_hat      = c_k*x0_plan + e_k;
        u_k        = utils.computeControl(x_hat, s_k, p_k, B, Ri, alpha, N, m, g);
        u_sol(:,k) = u_k;

        x_sol(:,k+1) = x_sol(:,k) + dt * (A*x_sol(:,k) + B*u_k);

        tau = tau + 1;
    end

    TATD_sol = utils.computeTATD(x_sol, xd, N, n_steps, dt);
end