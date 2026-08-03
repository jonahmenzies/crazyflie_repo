#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

struct params{
	// Physical Params
	const double g = 9.81;
	const double mass = 0.027;
	
	// drone physical characteristics
	const double a_theta = 9.81;
	const double a_phi = 9.81;
	const double a_r = 1;
	const double wij = 5;
	
	// Team Parameters
	static constexpr int N = 5; // No. Drones
	int q = 5; // weight of trajectory tracking for leader
	std::array<double, N> alpha = {0.2, 0.2, 0.2, 0.2, 0.2};
	
	// Timings
	double dt = 0.01; 
	
	// Trajectory
	double r_min = 0.15;
	double v_mean = 0.2;
	
	// waypoints // TODO: Move to pathManager.cpp at some point
	int numWaypoints = 10;
	Eigen::MatrixXd waypoints;
	Eigen::MatrixXd x; // Drone states
	Eigen::MatrixXd Ai; 
	Eigen::MatrixXd bi;
	Eigen::MatrixXd E_N = Eigen::MatrixXd::Identity(N,N);
	Eigen::MatrixXd E_n = Eigen::MatrixXd::Identity(10,10);
	std::vector<double> R_vals = {5600, 1005, 1050, 1050, 2920};
	std::vector<Eigen::MatrixXd> Ri;
	Eigen::MatrixXd D;
	Eigen::MatrixXd W;
	Eigen::MatrixXd A;
	Eigen::MatrixXd B;
	// constructor
	params(){
		waypoints.resize(numWaypoints, 3);
		waypoints.setZero();
		setWaypoint(0, 0.0, 0.0, 1.0);
		setWaypoint(1, 1.0, 0.0, 1.0);
		setWaypoint(2, 1.0, 1.0, 2.0);
		setWaypoint(3, 2.0, 1.0, 1.0);
		setWaypoint(4, 2.0, 0.0, 1.0);
		setWaypoint(5, 3.5, 0.5, 1.0);
		setWaypoint(6, 3.5, 1.5, 3.5);
		setWaypoint(7, 2.5, 2.5, 3.5);
		setWaypoint(8, 3.0, 3.5, 2.5);
		setWaypoint(9, 2.0, 4.0, 1.5);

		x.resize(N,10);
		x.setZero();
		setDroneState(0, -1.0, -1.0,  0.0,  0.0,  0.0,  0.0,  1.0, -1.0,  1.0,  0.0);
		setDroneState(1,  1.0,  0.0,  0.0,  2.0, -1.0,  0.0,  3.0,  0.0,  0.5,  1.0);
		setDroneState(2,  0.0,  3.0,  3.0,  4.0, -2.0,  0.0,  0.0,  3.0,  1.0,  1.0);
		setDroneState(3,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0);
		setDroneState(4, -1.0,  1.0,  1.0, -1.0,  1.0,  1.0,  2.0,  0.0, -0.5,  0.0);

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
		bi(7, 2) = -1/mass;
		bi(9, 3) = -a_r;

		A = Eigen::kroneckerProduct(E_N, Ai);
		B = Eigen::kroneckerProduct(E_N, bi);
		
		Ri.resize(N);
		for (int i = 0; i < N; i++){
			Ri[i] = R_vals[i] * Eigen::MatrixXd::Identity(4,4);
		}
		D.resize(5,4);
		D.setZero();
		D(0, 0) = 1;
		D(0, 1) = 1;
		D(1, 0) = -1;
		D(1, 2) = 1;
		D(2, 0) = -1;
		D(2, 3) = 1;
		D(3, 1) = -1;
		D(4, 2) = -1;
		
		W = wij * D.cwiseAbs();
	}

	// Functions
	void setWaypoint(int wpNo, double x_pos, double y_pos, double z_pos){
		waypoints(wpNo, 0) = x_pos;
		waypoints(wpNo, 1) = y_pos;
		waypoints(wpNo, 2) = z_pos;
	}

	void setDroneState(int drone, double x_pos, double x_d, double theta, double y_pos, double y_d, double phi, double z_pos, double z_d, double psi, double r){
		x(drone, 0) = x_pos;
		x(drone, 1) = x_d;
		x(drone, 2) = theta;
		x(drone, 3) = y_pos;
		x(drone, 4) = y_d;
		x(drone, 5) = phi;
		x(drone, 6) = z_pos;
		x(drone, 7) = z_d;
		x(drone, 8) = psi;
		x(drone, 9) = r;
	}
}
;
