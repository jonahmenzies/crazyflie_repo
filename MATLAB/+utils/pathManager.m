function [xd, tT] = pathManager(waypoints, formation_offsets, v_mean, dt, r_min, g)
%PATHMANAGER  Generate desired state trajectory from waypoints.
%   Smooths corners with circular arcs (radius r_min), interpolates to
%   uniform speed v_mean, and computes demanded attitude from acceleration.
%   Returns xd (n_steps x 10*N) and mission time tT.

path = waypoints(1,:);

for i = 2:size(waypoints,1)-1
    A_wp = waypoints(i-1,:);
    B_wp = waypoints(i,  :);
    C_wp = waypoints(i+1,:);

    AB = B_wp - A_wp;
    BC = C_wp - B_wp;

    unit_AB = AB / norm(AB);
    unit_BC = BC / norm(BC);

    cos_sigma   = max(-1, min(1, dot(unit_AB, unit_BC)));
    sigma       = acos(cos_sigma);
    delta_sigma = pi - sigma;

    n_vec = cross(unit_AB, unit_BC);
    if norm(n_vec) < 1e-10
        path = [path; B_wp];
        continue;
    end
    n_vec = n_vec / norm(n_vec);

    perp_AB = cross(n_vec, unit_AB);
    perp_AB = perp_AB / norm(perp_AB);

    R      = r_min;
    r_real = R / tan(delta_sigma / 2);
    r_real = min(r_real, 0.45 * min(norm(AB), norm(BC)));
    R      = r_real * tan(delta_sigma / 2);
    l      = B_wp - r_real * unit_AB;
    center = l + R * perp_AB;

    n_arc          = max(2, round(sigma * R / (v_mean * dt)));
    angle_per_step = sigma / n_arc;    
    e1             = (l - center) / norm(l - center);

    path = [path; l];
    for k = 1:n_arc
        theta = k * angle_per_step;
        point = center + R * rodrigues(e1, n_vec, theta);
        path  = [path; point];
    end
end

path = [path; waypoints(end,:)];

dists     = cumsum([0, sqrt(sum(diff(path).^2, 2))']);
tT        = dists(end) / v_mean;
s_uniform = 0:v_mean*dt:dists(end);
pos       = interp1(dists, path, s_uniform, 'linear', 'extrap');
T         = size(pos, 1);

vel            = diff(pos) / dt;
vel(end+1,:)   = vel(end,:);
accel          = diff(vel) / dt;
accel(end+1,:) = accel(end,:);

theta_d = accel(:,1) / g;
phi_d   = -accel(:,2) / g;

xd_central        = zeros(T, 10);
xd_central(:,1)   = pos(:,1);
xd_central(:,2)   = vel(:,1);
xd_central(:,3)   = theta_d;
xd_central(:,4)   = pos(:,2);
xd_central(:,5)   = vel(:,2);
xd_central(:,6)   = phi_d;
xd_central(:,7)   = pos(:,3);
xd_central(:,8)   = vel(:,3);
xd_central(:,9)   = 0;
xd_central(:,10)  = 0;

N_drones = size(formation_offsets, 1);
xd       = zeros(T, 10*N_drones);
for i = 1:N_drones
    idx           = (i-1)*10 + 1 : i*10;
    xd(:, idx)    = xd_central;
    xd(:, idx(1)) = xd_central(:,1) + formation_offsets(i,1);
    xd(:, idx(4)) = xd_central(:,4) + formation_offsets(i,2);
    xd(:, idx(7)) = xd_central(:,7) + formation_offsets(i,3);
end
end

function v_rot = rodrigues(v, axis, theta)
v_rot = v*cos(theta) + cross(axis,v)*sin(theta) + axis*dot(axis,v)*(1-cos(theta));
end