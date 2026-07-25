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
