// controller.h
#pragma once
#include "params.h"
#include "solver.h"

// ============================================================
// Controller — pure function, no mode logic
// ============================================================

struct Controller {
    const SystemParams* sys;
    const SolverResult* sol;
    const Config* cfg;
};

/// @brief Initialise controller
Controller ctrl_init(const SystemParams& sys, const SolverResult& sol,
                     const Config& cfg);

/// @brief Compute commands for all drones given current state and time
/// Doesn't care where x came from — measured or estimated
/// @param ctrl Controller with system matrices and precomputed tables
/// @param x Current state vector (N*n x 1)
/// @param t Current mission time (seconds)
/// @return Vector of N control inputs, each (4x1): theta_d, phi_d, deltaF_d, r_d
std::vector<VectorXd> ctrl_update(const Controller& ctrl,
                                   const VectorXd& x,
                                   double t);

/// @brief Clamp control outputs to physical limits
VectorXd clamp_inputs(const VectorXd& u);

/// @brief Compute TATD over recorded state history (eq 11)
double compute_tatd(const MatrixXd& x_history,
                    const MatrixXd& xd_history,
                    int num_drones, int skip_steps);
