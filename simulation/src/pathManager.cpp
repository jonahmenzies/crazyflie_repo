#include "pathManager.hpp"
#include "Eigen/Core"
#include <vector>

Trajectory pathManager(const params &P) {
    const Eigen::Index n = P.waypoints.rows();
    std::vector<Eigen::Vector3d> path;
    path.reserve(n);
    path.push_back(P.waypoints.row(0).transpose());

    for (Eigen::Index i = 1; i + 1 < n; ++i) {
        const Eigen::Vector3d A_wp = P.waypoints.row(i - 1).transpose();
        const Eigen::Vector3d B_wp = P.waypoints.row(i).transpose();
        const Eigen::Vector3d C_wp = P.waypoints.row(i + 1).transpose();

        const Eigen::Vector3d unit_AB = (B_wp - A_wp).normalized();
        const Eigen::Vector3d unit_BC = (C_wp - B_wp).normalized();

        const double cos_sigma = std::clamp(unit_AB.dot(unit_BC), -1.0, 1.0);
        const double sigma = std::acos(cos_sigma);
	
	const Eigen::Vector3d n_vec = 
    }
}
