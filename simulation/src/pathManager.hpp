#pragma once

#include <array>
#include <vector>
#include <Eigen/Dense>
#include "params.hpp"

struct Trajectory {
	Eigen::MatrixXd xd;
	double tT;
};

Trajectory pathManager(const params& P);
