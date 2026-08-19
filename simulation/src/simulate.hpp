#pragma once

#include <Eigen/Dense>
#include "params.hpp"
#include "riccatiSolver.hpp"

struct SimResult {
	Eigen::MatrixXd x;  // 10N x n_steps
	Eigen::MatrixXd u;  //  4N x n_steps
};

SimResult closedLoop(const params& P, const RiccatiSolution& ric);
SimResult openLoop(const params& P, const RiccatiSolution& ric);
SimResult semiOpenLoop(const params& P, const RiccatiSolution& ric, double replan_interval);
