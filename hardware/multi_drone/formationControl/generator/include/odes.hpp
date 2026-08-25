#pragma once
#include "riccatiSolver.hpp"
#include <Eigen/Dense>

// dS/dt — integrated backward from tT
Eigen::MatrixXd sODE(const Eigen::MatrixXd& S,
                     const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& AQ_sum,
                     const Eigen::MatrixXd& BR_sum);

// dp/dt — integrated backward from tT
Eigen::VectorXd pODE(const Eigen::VectorXd& p,
                     const Eigen::MatrixXd& S,
                     const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& AQ_sum,
                     const Eigen::MatrixXd& BR_sum,
                     const Eigen::VectorXd& xd_t);

// dC/dt — integrated forward from replan time
Eigen::MatrixXd cODE(const Eigen::MatrixXd& C,
                     const Eigen::MatrixXd& S,
                     const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& BR_sum);

// de/dt — integrated forward from replan time
Eigen::VectorXd eODE(const Eigen::VectorXd& e,
                     const Eigen::MatrixXd& S,
                     const Eigen::VectorXd& p,
                     const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& BR_sum);

void stepCE(Eigen::MatrixXd& C, Eigen::VectorXd& e,
            const RiccatiSolution& ric, const params& P, int k);
