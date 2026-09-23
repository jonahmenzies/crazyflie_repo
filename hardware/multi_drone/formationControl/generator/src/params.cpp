#include "params.hpp"
#include <unsupported/Eigen/KroneckerProduct>

params::params(){
	// TODO: Move to pathManager.cpp
	// Rectangular circuit, corners plus edge midpoints, with one smooth
	// altitude cycle per lap. xy is a plain loop so the only thing varying
	// interestingly is z.
	//
	//   z = 0.50 + 0.225 * (1 - cos(2*pi*i/8))
	//
	// Low point at waypoint 0, peak at waypoint 4 (diagonally opposite),
	// continuous across the 7 -> 0 wrap so the loop closes without an
	// altitude step. Envelope allows waypoints in x [-0.40, 0.40],
	// y [-0.33, 0.33], z [0.40, 1.03] once the formation offsets
	// (±0.45 x, ±0.32 y) are added.
	waypoints.resize(numWaypoints, 3);
	waypoints.setZero();
	setWaypoint(0, -0.35, -0.28, 0.50);   // corner, low point
	setWaypoint(1, -0.35,  0.00, 0.57);
	setWaypoint(2, -0.35,  0.28, 0.73);   // corner
	setWaypoint(3,  0.00,  0.28, 0.88);
	setWaypoint(4,  0.35,  0.28, 0.95);   // corner, peak
	setWaypoint(5,  0.35,  0.00, 0.88);
	setWaypoint(6,  0.35, -0.28, 0.73);   // corner
	setWaypoint(7,  0.00, -0.28, 0.57);

	// Four corners of the rectangle. The centre drone (formerly slot 4,
	// offset 0,0,+0.12) is gone.
	formation_offsets.resize(N,3);
	formation_offsets <<  0.45,  0.32,  0.00,
	                     -0.45,  0.32,  0.00,
	                      0.45, -0.32,  0.00,
	                     -0.45, -0.32,  0.00;

	x0.resize(N,10);
	x0.setZero();
	setDroneState(0,   0.18, 0.0, 0.0,   0.00, 0.0, 0.0,  0.55, 0.0, 0.0, 0.0);
	setDroneState(1,  -0.72, 0.0, 0.0,   0.14, 0.0, 0.0,  0.65, 0.0, 0.0, 0.0);
	setDroneState(2,   0.03, 0.0, 0.0,  -0.50, 0.0, 0.0,  0.62, 0.0, 0.0, 0.0);
	setDroneState(3,  -0.75, 0.0, 0.0,  -0.62, 0.0, 0.0,  0.54, 0.0, 0.0, 0.0);

	systemDynamics();
	controlWeight();
	stateWeight();

	// Stored sums for efficiency
	// TODO: Move to ricattiSolver.cpp
	BR_sum.resize(10*N,10*N);
	BR_sum.setZero();
	AQ_sum.resize(10*N,10*N);
	AQ_sum.setZero();
	for(int i = 0; i <N; i++){
		Eigen::MatrixXd Bi = B.middleCols(i*4, 4);
		AQ_sum += alpha[i] * Q[i];
		BR_sum += (1.0/alpha[i]) * Bi * Ri[i].inverse() * Bi.transpose();
	}
}

// Functions
void params::setWaypoint(int wpNo, double x_pos, double y_pos, double z_pos){
	waypoints(wpNo, 0) = x_pos;
	waypoints(wpNo, 1) = y_pos;
	waypoints(wpNo, 2) = z_pos;
}

void params::setDroneState(int drone, double x_pos, double x_d, double theta, double y_pos, double y_d, double phi, double z_pos, double z_d, double psi, double r){
	x0(drone, 0) = x_pos;
	x0(drone, 1) = x_d;
	x0(drone, 2) = theta;
	x0(drone, 3) = y_pos;
	x0(drone, 4) = y_d;
	x0(drone, 5) = phi;
	x0(drone, 6) = z_pos;
	x0(drone, 7) = z_d;
	x0(drone, 8) = psi;
	x0(drone, 9) = r;
}
void params::systemDynamics(){
	Ai.resize(10,10);
	Ai.setZero();
	Ai(0, 1) = 1;
	Ai(1, 2) = g;
	Ai(2, 2) = -a_theta;
	Ai(3, 4) = 1;
	Ai(4, 5) = -g;
	Ai(5, 5) = -a_phi;
	Ai(6, 7) = 1;
	Ai(8, 9) = 1;
	Ai(9, 9) = -a_r;

	bi.resize(10,4);
	bi.setZero();
	bi(2, 0) = a_theta;
	bi(5, 1) = a_phi;
	bi(7, 2) = -1.0/mass;
	bi(9, 3) = -a_r;

	A = Eigen::kroneckerProduct(E_N, Ai);
	B = Eigen::kroneckerProduct(E_N, bi);
}
void params::stateWeight(){
	// Incidence matrix, (agents x edges). Spanning tree hubbed at agent 0:
	//   edge 0: 0 - 1
	//   edge 1: 0 - 2
	//   edge 2: 0 - 3
	// This is what the five-agent graph reduces to once agent 4 (and the
	// 1-4 edge) is removed.
	D.resize(4,3);
	D.setZero();
	D(0, 0) =  1;
	D(1, 0) = -1;
	D(0, 1) =  1;
	D(2, 1) = -1;
	D(0, 2) =  1;
	D(3, 2) = -1;

	W = wij * D.cwiseAbs();
	D_hat = Eigen::kroneckerProduct(D,E_n);
	Q.resize(N);
	W_hat.resize(N);
	for(int i = 0; i < N; i++){
		Eigen::MatrixXd Wi = W.row(i).asDiagonal();
		W_hat[i] = Eigen::kroneckerProduct(Wi, E_n);
		Q[i] = D_hat * W_hat[i] * D_hat.transpose();
	}

	q_diag.resize(10*N,10*N);
	q_diag.setZero();
	int row_start = leader * 10;
	q_diag.block(row_start, row_start, 10, 10) = q * E_n;
	Q[leader] += q_diag;
}
void params::controlWeight(){
	Ri.resize(N);
	for(int i = 0; i < N; i++){
		Ri[i] = R_vals[i] * Eigen::MatrixXd::Identity(4,4);
	}
}
