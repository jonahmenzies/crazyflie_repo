#pragma once

#include <Eigen/Dense>
#include "params.hpp"

struct Trajectory {
	Eigen::MatrixXd xd;  // T x 10*N desired states
	double tT;           // mission time [s]
};

Trajectory pathManager(const params& P);
