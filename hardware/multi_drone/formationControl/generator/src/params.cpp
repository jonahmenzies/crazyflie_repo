#include "params.hpp"
#include <cstdio>
#include <cstdlib>
#include <unsupported/Eigen/KroneckerProduct>

params::params(){
	// ---- Waypoints ----
	// Path of the formation centre, one {x, y, z} per waypoint.
	// The number of waypoints comes from this list.
	const std::vector<std::array<double, 3>> wp = {
		{-0.35, -0.28, 0.50},   // 0  corner, low point
		{-0.35,  0.00, 0.57},   // 1
		{-0.35,  0.28, 0.73},   // 2  corner
		{ 0.00,  0.28, 0.88},   // 3
		{ 0.35,  0.28, 0.95},   // 4  corner, peak
		{ 0.35,  0.00, 0.88},   // 5
		{ 0.35, -0.28, 0.73},   // 6  corner
		{ 0.00, -0.28, 0.57},   // 7
	};

	// Stop if there are too few waypoints, or two in a row are the same point
	if(wp.size() < 2){
		std::fprintf(stderr, "ERROR: need at least 2 waypoints\n");
		std::exit(1);
	}
	for(size_t i = 1; i < wp.size(); i++){
		if(wp[i] == wp[i-1]){
			std::fprintf(stderr, "ERROR: waypoints %zu and %zu are the same point\n", i-1, i);
			std::exit(1);
		}
	}

	waypoints.resize(wp.size(), 3);
	for(size_t i = 0; i < wp.size(); i++){
		waypoints(i, 0) = wp[i][0];
		waypoints(i, 1) = wp[i][1];
		waypoints(i, 2) = wp[i][2];
	}

	// ---- Formation ----
	// Each drone's offset from the centre (x, y, z): four corners of a rectangle
	formation_offsets.resize(N, 3);
	formation_offsets <<  0.45,  0.32,  0.00,
	                     -0.45,  0.32,  0.00,
	                      0.45, -0.32,  0.00,
	                     -0.45, -0.32,  0.00;

	// ---- Dynamics and weights ----
	systemDynamics();
	controlWeight();
	stateWeight();

	// ---- Sums used by the Riccati solver ----
	BR_sum.resize(10*N, 10*N);
	BR_sum.setZero();
	AQ_sum.resize(10*N, 10*N);
	AQ_sum.setZero();
	for(int i = 0; i < N; i++){
		Eigen::MatrixXd Bi = B.middleCols(i*4, 4);
		AQ_sum += alpha[i] * Q[i];
		BR_sum += (1.0/alpha[i]) * Bi * Ri[i].inverse() * Bi.transpose();
	}
}

// Linear model of one drone, then repeated for the whole swarm.
// State per drone:   [x, vx, pitch, y, vy, roll, z, vz, yaw, yaw rate]
// Control per drone: [pitch cmd, roll cmd, u_z, yaw rate cmd]
void params::systemDynamics(){
	Ai.resize(10, 10);
	Ai.setZero();
	Ai(0, 1) = 1;           // x' = vx
	Ai(1, 2) = g;           // vx' = g * pitch
	Ai(2, 2) = -a_theta;    // pitch follows its command
	Ai(3, 4) = 1;           // y' = vy
	Ai(4, 5) = -g;          // vy' = -g * roll
	Ai(5, 5) = -a_phi;      // roll follows its command
	Ai(6, 7) = 1;           // z' = vz
	Ai(8, 9) = 1;           // yaw' = yaw rate
	Ai(9, 9) = -a_r;        // yaw rate follows its command

	bi.resize(10, 4);
	bi.setZero();
	bi(2, 0) = a_theta;     // pitch command
	bi(5, 1) = a_phi;       // roll command
	bi(7, 2) = -1.0/mass;   // vz' = -u_z / mass
	bi(9, 3) = -a_r;        // yaw rate command

	// Same model for every drone, down the diagonal
	A = Eigen::kroneckerProduct(E_N, Ai);
	B = Eigen::kroneckerProduct(E_N, bi);
}

// State weight for each drone, from the communication graph
void params::stateWeight(){
	// Incidence matrix (drones x edges). Star around drone 0:
	//   edge 0: 0 - 1
	//   edge 1: 0 - 2
	//   edge 2: 0 - 3
	D.resize(4, 3);
	D.setZero();
	D(0, 0) =  1;
	D(1, 0) = -1;
	D(0, 1) =  1;
	D(2, 1) = -1;
	D(0, 2) =  1;
	D(3, 2) = -1;

	// Each drone is weighted on the edges it is part of
	W = wij * D.cwiseAbs();
	D_hat = Eigen::kroneckerProduct(D, E_n);
	Q.resize(N);
	W_hat.resize(N);
	for(int i = 0; i < N; i++){
		Eigen::MatrixXd Wi = W.row(i).asDiagonal();
		W_hat[i] = Eigen::kroneckerProduct(Wi, E_n);
		Q[i] = D_hat * W_hat[i] * D_hat.transpose();
	}

	// Only the leader is weighted on tracking the path
	q_diag.resize(10*N, 10*N);
	q_diag.setZero();
	int row_start = leader * 10;
	q_diag.block(row_start, row_start, 10, 10) = q * E_n;
	Q[leader] += q_diag;
}

// Control effort weight for each drone
void params::controlWeight(){
	Ri.resize(N);
	for(int i = 0; i < N; i++){
		Ri[i] = R_vals[i] * Eigen::MatrixXd::Identity(4, 4);
	}
}
