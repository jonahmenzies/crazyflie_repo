    function TATD = computeTATD(x, xd, N, n_steps, dt)
%COMPUTETATD  Time-Averaged Tracking Deviation (m).
%   Implements Eq. (11) from Jiang, Gonzalez & McFadyen (2020).
%   Skips the first 5 seconds to exclude initial transient.
%
%   Inputs:
%     x       -- state trajectory   (10N x n_steps)
%     xd      -- desired trajectory (n_steps x 10N)
%     N       -- number of agents
%     n_steps -- total simulation steps
%     dt      -- timestep (s)
%
%   Output:
%     TATD -- scalar RMS tracking deviation (m)

skip = round(5 / dt);
NT   = n_steps - skip;

pos_idx = [];
for i = 1:N
    base    = (i-1) * 10;
    pos_idx = [pos_idx, base+1, base+4, base+7];
end

err  = x(pos_idx, skip:end) - xd(skip:end, pos_idx)';
TATD = sqrt(sum(err(:).^2) / (N * NT));

end