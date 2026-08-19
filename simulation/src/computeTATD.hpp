#pragma once

#include <Eigen/Dense>
#include "params.hpp"

// TATD (Jiang Eq. 11) over the final `frac` of the simulation
double computeTATD20(const Eigen::MatrixXd& x,
                     const Eigen::MatrixXd& xd,
                     const params& P,
                     double frac = 0.2);

// TATD at each individual timestep (no windowing)
Eigen::VectorXd computeTATDInstantaneous(const Eigen::MatrixXd& x,
                                         const Eigen::MatrixXd& xd,
                                         const params& P);
