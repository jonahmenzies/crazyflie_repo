// controller.cpp
#include "controller.h"
#include <cmath>
#include <algorithm>

// ============================================================
// Internal
// ============================================================

/// @brief Evaluate control law eq 9a for one drone
/// u_i = -(1/alpha_i) * R_i^-1 * B_i^T * (s * x + p)
VectorXd compute_ui(const Controller& ctrl, const VectorXd& x,
                    int timestep, int drone_i) {
    const MatrixXd& s = ctrl.sol->s_table[timestep];
    const VectorXd& p = ctrl.sol->p_table[timestep];

    VectorXd correction = s * x + p;

    double alpha_i = ctrl.cfg->alpha(drone_i);
    VectorXd u_i = -(1.0 / alpha_i)
                   * ctrl.sys->R[drone_i].inverse()
                   * ctrl.sys->B[drone_i].transpose()
                   * correction;

    return clamp_inputs(u_i);
}

// ============================================================
// Public
// ============================================================

Controller ctrl_init(const SystemParams& sys, const SolverResult& sol,
                     const Config& cfg) {
    Controller ctrl;
    ctrl.sys = &sys;
    ctrl.sol = &sol;
    ctrl.cfg = &cfg;
    return ctrl;
}

std::vector<VectorXd> ctrl_update(const Controller& ctrl,
                                   const VectorXd& x,
                                   double t) {
    int timestep = static_cast<int>(t / ctrl.cfg->dt);

    // Clamp timestep to valid range
    if (timestep < 0) timestep = 0;
    if (timestep > ctrl.sol->num_steps) timestep = ctrl.sol->num_steps;

    std::vector<VectorXd> u_out(ctrl.cfg->num_drones);
    for (int i = 0; i < ctrl.cfg->num_drones; i++) {
        u_out[i] = compute_ui(ctrl, x, timestep, i);
    }
    return u_out;
}

VectorXd clamp_inputs(const VectorXd& u) {
    VectorXd clamped = u;
    // u(0) = theta_d (pitch command)
    clamped(0) = std::clamp(clamped(0), -MAX_PITCH, MAX_PITCH);
    // u(1) = phi_d (roll command)
    clamped(1) = std::clamp(clamped(1), -MAX_ROLL, MAX_ROLL);
    // u(2) = deltaF_d (thrust command) — no negative thrust
    // Paper limits to actual drone power, we just prevent negative
    if (clamped(2) < 0.0) clamped(2) = 0.0;
    // u(3) = r_d (yaw rate command)
    clamped(3) = std::clamp(clamped(3), -MAX_YAW_RATE, MAX_YAW_RATE);
    return clamped;
}

double compute_tatd(const MatrixXd& x_history,
                    const MatrixXd& xd_history,
                    int num_drones, int skip_steps) {
    int n = STATES_PER_DRONE;
    int total_steps = x_history.cols() - skip_steps;
    double sum = 0.0;

    for (int k = skip_steps; k < x_history.cols(); k++) {
        for (int i = 0; i < num_drones; i++) {
            // Extract position states only: x(0), y(3), z(6) for drone i
            int base = i * n;
            double dx = x_history(base + 0, k) - xd_history(base + 0, k);
            double dy = x_history(base + 3, k) - xd_history(base + 3, k);
            double dz = x_history(base + 6, k) - xd_history(base + 6, k);
            sum += dx * dx + dy * dy + dz * dz;
        }
    }

    return std::sqrt(sum / (num_drones * total_steps));
}
