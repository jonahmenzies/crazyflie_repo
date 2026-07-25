function dsdt = riccatiODE(~, sv, A, sum_alpha_Q, sum_BR, N)
S    = reshape(sv, 10*N, 10*N);
dS   = -S*A - A'*S - sum_alpha_Q + S*sum_BR*S;
dsdt = dS(:);
end