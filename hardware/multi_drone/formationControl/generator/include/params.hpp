#pragma once

#include <array>
#include <vector>
#include <Eigen/Dense>

struct params{
	// ---- Physical constants ----
	const double g = 9.81;          // gravity, m/s^2
	const double mass = 0.027;      // drone mass, kg
	const double a_theta = 7.41;    // pitch response rate, from attitude_id.py
	const double a_phi = 7.49;      // roll response rate, from attitude_id.py
	const double a_r = 2.67;        // yaw rate response rate, from yawrate_id.py

	// ---- Controller settings ----
	static constexpr int N = 4;     // number of drones
	const double wij = 5;           // edge weight between connected drones
	const int q = 5;                // leader's trajectory tracking weight
	const double r_min = 0.15;      // corner turn radius, m
	const double v_mean = 0.2;      // speed along the path, m/s
	const double dt = 0.01;         // timestep, s

	int leader = 0;                                       // leader drone index
	std::array<double, N> alpha = {0.2, 0.2, 0.2, 0.2};  // team cost weight per drone
	std::vector<double> R_vals = {900, 900, 900, 900};   // control effort weight per drone
	std::vector<Eigen::MatrixXd> Ri;                      // control weight matrices, built from R_vals

	// ---- Path and formation ----
	Eigen::MatrixXd waypoints;          // formation centre path, one row (x, y, z) per waypoint
	Eigen::MatrixXd formation_offsets;  // each drone's offset from the centre, N x 3

	// ---- Dynamics ----
	Eigen::MatrixXd Ai;     // one drone, 10 x 10
	Eigen::MatrixXd bi;     // one drone, 10 x 4
	Eigen::MatrixXd A;      // whole swarm
	Eigen::MatrixXd B;      // whole swarm

	// ---- Communication graph and state weights ----
	Eigen::MatrixXd D;                  // incidence matrix, drones x edges
	Eigen::MatrixXd D_hat;
	Eigen::MatrixXd W;
	std::vector<Eigen::MatrixXd> W_hat;
	std::vector<Eigen::MatrixXd> Q;     // state weight per drone
	Eigen::MatrixXd q_diag;             // leader's tracking weight

	// ---- Identity helpers ----
	Eigen::MatrixXd E_N = Eigen::MatrixXd::Identity(N, N);
	Eigen::MatrixXd E_n = Eigen::MatrixXd::Identity(10, 10);

	// ---- Sums used by the Riccati solver ----
	Eigen::MatrixXd BR_sum;
	Eigen::MatrixXd AQ_sum;

	params();

	void systemDynamics();
	void stateWeight();
	void controlWeight();
};
