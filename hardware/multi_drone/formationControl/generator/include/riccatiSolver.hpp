#pragma once

#include <vector>
#include <Eigen/Dense>
#include "params.hpp"

// S(t) and p(t) at every timestep, plus the trajectory they were solved for
struct RiccatiSolution {
	std::vector<Eigen::MatrixXd> s_store;  // S at each timestep
	std::vector<Eigen::VectorXd> p_store;  // p at each timestep
	Eigen::MatrixXd xd;                    // desired state, n_steps x 10N
	double tT;                             // mission time, s
	int n_steps;                           // number of timesteps
};

RiccatiSolution riccatiSolver(const params& P);
