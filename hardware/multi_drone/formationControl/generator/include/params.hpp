#pragma once

#include <array>
#include <vector>
#include <Eigen/Dense>

struct params{
	// Physical Constants
	const double g = 9.81;
	const double mass = 0.027;
	const double a_theta = 7.41;
	const double a_phi = 7.49;
	const double a_r = 2.67;
	
	// Thesis Specific Constants
	const double wij = 5; // Edge Weight
	static constexpr int N = 4; // No. Drones
	const int q = 5; // weight of trajectory tracking for leader
	const double r_min = 0.15;
	const double v_mean = 0.2;
	const double dt = 0.01; 
	const double t0 = 0.0;

	// Optimisable parameters
	int leader = 0; // Drone leader index
	std::array<double, N> alpha = {0.2, 0.2, 0.2, 0.2};
	std::vector<double> R_vals = {900, 900, 900, 900};
	std::vector<Eigen::MatrixXd> Ri;

	// waypoints // TODO: Move to pathManager.cpp at some point
	int numWaypoints = 14;
	Eigen::MatrixXd waypoints;
	Eigen::MatrixXd x0; // Drone states

	// System Dynamics
	Eigen::MatrixXd Ai; 
	Eigen::MatrixXd bi;
	Eigen::MatrixXd A;
	Eigen::MatrixXd B;
	
	// Communication
	std::vector<Eigen::MatrixXd> Q;
	Eigen::MatrixXd D;
	Eigen::MatrixXd D_hat;
	Eigen::MatrixXd W;
	std::vector<Eigen::MatrixXd> W_hat;
	Eigen::MatrixXd q_diag;
	Eigen::MatrixXd formation_offsets;  // N x 3

	// Helper Matrices
	Eigen::MatrixXd E_N = Eigen::MatrixXd::Identity(N,N);
	Eigen::MatrixXd E_n = Eigen::MatrixXd::Identity(10,10);

	// Efficient storage
	Eigen::MatrixXd BR_sum;
	Eigen::MatrixXd AQ_sum;

	// constructor
	params();

	// Functions
	void setWaypoint(int wpNo, double x_pos, double y_pos, double z_pos);
	void setDroneState(int drone, double x_pos, double x_d, double theta, double y_pos, double y_d, double phi, double z_pos, double z_d, double psi, double r);
	void systemDynamics();
	void stateWeight();
	void controlWeight();

	Eigen::VectorXd x0_flat() const {
		Eigen::MatrixXd tmp = x0.transpose();  // 10 x N
		return Eigen::Map<const Eigen::VectorXd>(tmp.data(), tmp.size());
	}
}
;

