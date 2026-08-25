#pragma once

#include <vector>
#include <Eigen/Dense>
#include "params.hpp"

struct RiccatiSolution {
	std::vector<Eigen::MatrixXd> s_store;  // S(t_k), k = 0 .. n_steps-1
	std::vector<Eigen::VectorXd> p_store;  // p(t_k)
	Eigen::MatrixXd xd;
	double tT;
	int n_steps;
};

RiccatiSolution riccatiSolver(const params& P);
