#include <iostream>
#include <Eigen/Dense>

struct params{
	// Physical Params
	const double g = 9.81;
	const double mass = 0.027;
	
	// drone physical characteristics
	const double a_theta = 9.81;
	const double a_phi = 9.81;
	const double a_r = 1;

	// Team Parameters
	const int N = 5; // No. Drones
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
	
	// constructor
	params(){
		waypoints.resize(numWaypoints, 3);
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
	}

	// Functions
	void setWaypoint(int wpNo, double x, double y, double z){
		waypoints(wpNo, 0) = x;
		waypoints(wpNo, 1) = y;
		waypoints(wpNo, 2) = z;
	}
};
