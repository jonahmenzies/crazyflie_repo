function dedt = eODE(t, e, t_s, s_store, t_p, p_store, A, sum_BR, N)
%EODE  de/dt = (A - sum_BR*S(t))*e - sum_BR*p(t)
S    = utils.lookupMatrix(t, t_s, s_store);
p    = interp1(t_p, p_store, t, 'linear', 'extrap')';
dedt = (A - sum_BR*S)*e - sum_BR*p;
end