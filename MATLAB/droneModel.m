params = getParams();
g         = params.g;
a_theta   = params.a_theta;
a_phi     = params.a_phi;
a_r       = params.a_r;
m         = params.m;
N         = params.N;
leader    = 1;
q         = 5;

% 10x10 State matrix
Ai = [0,  1,        0,   0,  0,        0,   0,  0,  0,      0;
      0,  0,        g,   0,  0,        0,   0,  0,  0,      0;
      0,  0,  -a_theta,  0,  0,        0,   0,  0,  0,      0;
      0,  0,        0,   0,  1,        0,   0,  0,  0,      0;
      0,  0,        0,   0,  0,       -g,   0,  0,  0,      0;
      0,  0,        0,   0,  0,   -a_phi,   0,  0,  0,      0;
      0,  0,        0,   0,  0,        0,   0,  1,  0,      0;
      0,  0,        0,   0,  0,        0,   0,  0,  0,      0;
      0,  0,        0,   0,  0,        0,   0,  0,  0,      1;
      0,  0,        0,   0,  0,        0,   0,  0,  0,  -a_r];

% 10x4 Input matrix
bi = [0,        0,    0,     0;
      0,        0,    0,     0;
      a_theta,  0,    0,     0;
      0,        0,    0,     0;
      0,        0,    0,     0;
      0,     a_phi,   0,     0;
      0,        0,    0,     0;
      0,        0,  1/m,     0;
      0,        0,    0,     0;
      0,        0,    0,  a_r];

E_N = eye(N);
E_n = eye(10);
A   = kron(E_N, Ai);
B   = kron(E_N, bi);

D = [  1,   1,  0,  0;
      -1,   0,  1,  0;
      -1,   0,  0,  1;
       0,  -1,  0,  0;
       0,   0, -1,  0];

D_hat = kron(D, E_n);

W = [5, 5, 0, 0;
     5, 0, 5, 0;
     5, 0, 0, 5;
     0, 5, 0, 0;
     0, 0, 5, 0];

W_hat = cell(N, 1);
Q     = cell(N, 1);

for i = 1:N
    W_hat{i} = kron(diag(W(i,:)), E_n);
    Q{i}     = D_hat * W_hat{i} * D_hat';
end

% Add trajectory tracking term to leader's Q
q_diag                                       = zeros(10*N, 10*N);
row_start                                    = (leader-1)*10 + 1;
row_end                                      = leader*10;
q_diag(row_start:row_end, row_start:row_end) = q * E_n;
Q{leader}                                    = Q{leader} + q_diag;

Ri = cell(N, 1);
for i = 1:N
    Ri{i} = eye(4);   % 4x4, tune during optimisation
end


alpha = [0.2, 0.2, 0.2, 0.2, 0.2];   % from paper


function params = getParams()
    params.m       = 0.027;
    params.a_theta = 1;
    params.a_phi   = 1;
    params.a_r     = 1;
    params.g       = 9.81;
    params.N       = 5;
end