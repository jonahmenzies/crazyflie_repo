#include "pathManager.hpp"
#include "Eigen/Core"
#include <vector>

Trajectory pathManager(const params &P){
	std::vector<Eigen::MatrixXd> path;
	path[0] = P.waypoints.row(0);
	Eigen::MatrixXd A_wp;
	Eigen::MatrixXd B_wp;
	Eigen::MatrixXd C_wp;
	Eigen::MatrixXd AB;
	Eigen::MatrixXd BC;
	Eigen::MatrixXd unit_AB;
	Eigen::MatrixXd unit_BC;
	for(int i = 1; i < P.waypoints.rows(); i++){
		A_wp = P.waypoints.row(i-1);
		B_wp = P.waypoints.row(i);
		C_wp = P.waypoints.row(i+1);
	
		AB = B_wp - A_wp;
		BC = C_wp - B_wp;

		unit_AB = AB / AB.norm();
		unit_BC = BC / BC.norm();
		
		double cos_sigma = 
	}


}
