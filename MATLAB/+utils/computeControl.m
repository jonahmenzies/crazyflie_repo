function u = computeControl(x, S, p, B, Ri, alpha, N, m, g)
    sp = S*x + p;
    u  = zeros(4*N, 1);
    for i = 1:N
        Bi = B(:, (i-1)*4+1 : i*4);
        u((i-1)*4+1:i*4) = -(1/alpha(i)) * (Ri{i}\Bi') * sp;
    end
    u = inclinationProtection(u, N, m, g);
end


function u = inclinationProtection(u, N, m, g)
    for i = 1:N
        idx       = (i-1)*4 + 1 : i*4;
        u(idx(1)) = max(-pi/4, min(pi/4,  u(idx(1))));  % pitch
        u(idx(2)) = max(-pi/4, min(pi/4,  u(idx(2))));  % roll
        u(idx(3)) = max(-m*g,  min(2*m*g, u(idx(3))));  % thrust
        u(idx(4)) = max(-2*pi, min(2*pi,  u(idx(4))));  % yaw rate
    end
end