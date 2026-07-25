function dpdt = pODE(t, p, t_s, s_store, A, sum_alpha_Q, sum_BR, xd, dt, N)
S    = utils.lookupMatrix(t, t_s, s_store);
k    = max(1, min(size(xd,1), round(t/dt) + 1));
dpdt = (S*sum_BR - A')*p + sum_alpha_Q*xd(k,:)';
end
