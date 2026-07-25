function dcdt = cODE(t, cv, t_s, s_store, A, sum_BR, N)
S    = utils.lookupMatrix(t, t_s, s_store);
C    = reshape(cv, 10*N, 10*N);
dC   = (A - sum_BR*S) * C;
dcdt = dC(:);
end