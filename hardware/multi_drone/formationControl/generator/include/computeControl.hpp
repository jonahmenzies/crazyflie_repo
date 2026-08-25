#pragma once

#include <Eigen/Dense>
#include "params.hpp"

Eigen::VectorXd computeControl(const Eigen::VectorXd& x,
                               const Eigen::MatrixXd& S,
                               const Eigen::VectorXd& p,
                               const params& P);
