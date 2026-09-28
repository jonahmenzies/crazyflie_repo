#pragma once

#include <Eigen/Dense>
#include "params.hpp"

// Desired state of every drone at every timestep
struct Trajectory {
	Eigen::MatrixXd xd;  // n_steps x 10N
	double tT;           // mission time, s
};

Trajectory pathManager(const params& P);
