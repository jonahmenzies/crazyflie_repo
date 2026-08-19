#include "pathManager.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

// Rotate v about a unit axis by theta
static Eigen::Vector3d rodrigues(const Eigen::Vector3d& v, const Eigen::Vector3d& axis, double theta){
	return v*std::cos(theta)
	     + axis.cross(v)*std::sin(theta)
	     + axis*axis.dot(v)*(1.0 - std::cos(theta));
}

Trajectory pathManager(const params& P){
	const int n_wp = P.waypoints.rows();

	std::vector<Eigen::Vector3d> path;
	path.push_back(P.waypoints.row(0).transpose());

	for(int i = 1; i < n_wp - 1; i++){
		Eigen::Vector3d A_wp = P.waypoints.row(i-1).transpose();
		Eigen::Vector3d B_wp = P.waypoints.row(i  ).transpose();
		Eigen::Vector3d C_wp = P.waypoints.row(i+1).transpose();

		Eigen::Vector3d AB = B_wp - A_wp;
		Eigen::Vector3d BC = C_wp - B_wp;

		Eigen::Vector3d unit_AB = AB.normalized();
		Eigen::Vector3d unit_BC = BC.normalized();

		double cos_sigma   = std::clamp(unit_AB.dot(unit_BC), -1.0, 1.0);
		double sigma       = std::acos(cos_sigma);
		double delta_sigma = M_PI - sigma;

		Eigen::Vector3d n_vec = unit_AB.cross(unit_BC);
		if(n_vec.norm() < 1e-10){
			path.push_back(B_wp);
			continue;
		}
		n_vec.normalize();

		Eigen::Vector3d perp_AB = n_vec.cross(unit_AB).normalized();

		double R      = P.r_min;
		double r_real = R / std::tan(delta_sigma/2.0);
		r_real = std::min(r_real, 0.45 * std::min(AB.norm(), BC.norm()));
		R      = r_real * std::tan(delta_sigma/2.0);

		Eigen::Vector3d l      = B_wp - r_real*unit_AB;
		Eigen::Vector3d center = l + R*perp_AB;

		int n_arc = std::max(2, (int)std::lround(sigma*R/(P.v_mean*P.dt)));
		double angle_per_step = sigma / n_arc;
		Eigen::Vector3d e1 = (l - center).normalized();

		path.push_back(l);
		for(int k = 1; k <= n_arc; k++){
			double theta = k * angle_per_step;
			path.push_back(center + R*rodrigues(e1, n_vec, theta));
		}
	}
	path.push_back(P.waypoints.row(n_wp-1).transpose());
	std::vector<double> dists(path.size(), 0.0);
	for(size_t i = 1; i < path.size(); i++){
		dists[i] = dists[i-1] + (path[i] - path[i-1]).norm();
	}


	double ds = P.v_mean * P.dt;
	int T = (int)std::floor(dists.back()/ds) + 1;
	double tT = (T - 1) * P.dt;

	Eigen::MatrixXd pos(T,3);
	int k = 0;
	for(int t = 0; t < T; t++){
		double s = t * ds;
		while(k + 2 < (int)path.size() && dists[k+1] < s) k++;

		double seg = dists[k+1] - dists[k];
		double u   = (seg > 1e-12) ? (s - dists[k]) / seg : 0.0;
		pos.row(t) = (path[k] + u*(path[k+1] - path[k])).transpose();
	}
	Eigen::MatrixXd vel(T,3), accel(T,3);

	for(int t = 0; t < T-1; t++) vel.row(t) = (pos.row(t+1) - pos.row(t)) / P.dt;
	vel.row(T-1) = vel.row(T-2);

	for(int t = 0; t < T-1; t++) accel.row(t) = (vel.row(t+1) - vel.row(t)) / P.dt;
	accel.row(T-1) = accel.row(T-2);
	Eigen::MatrixXd xd_central = Eigen::MatrixXd::Zero(T,10);
	xd_central.col(0) =  pos.col(0);
	xd_central.col(1) =  vel.col(0);
	xd_central.col(2) =  accel.col(0) / P.g;   // theta_d
	xd_central.col(3) =  pos.col(1);
	xd_central.col(4) =  vel.col(1);
	xd_central.col(5) = -accel.col(1) / P.g;   // phi_d
	xd_central.col(6) =  pos.col(2);
	xd_central.col(7) =  vel.col(2);
	// cols 8,9 (psi, r) stay zero

	const int N_drones = P.formation_offsets.rows();
	Eigen::MatrixXd xd = Eigen::MatrixXd::Zero(T, 10*N_drones);
	for(int i = 0; i < N_drones; i++){
		int idx = i*10;
		xd.middleCols(idx,10) = xd_central;
		xd.col(idx+0).array() += P.formation_offsets(i,0);
		xd.col(idx+3).array() += P.formation_offsets(i,1);
		xd.col(idx+6).array() += P.formation_offsets(i,2);
	}

	return Trajectory{xd, tT};
}
