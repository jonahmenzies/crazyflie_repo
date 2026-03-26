// solver.cpp
#include "solver.h"
#include <fstream>

// ============================================================
// ODE right-hand sides
// ============================================================

/// @brief Equation 9b: rate of change of s
/// s_dot = -s*A - A^T*s - weighted_Q + s * BR_sum * s
/// All terms except s itself are constants from SystemParams
/// @param s Current value of s (N*n x N*n)
/// @param sys System matrices containing A, weighted_Q, BR_sum
/// @return Rate of change of s (N*n x N*n)
MatrixXd s_dot(const MatrixXd& s, const SystemParams& sys) {
    return -s * sys.A - sys.A.transpose() * s - sys.weighted_Q + s * sys.BR_sum * s;
}

/// @brief Equation 9c: rate of change of p
/// p_dot = weighted_Q * x_d + s * BR_sum * p
/// @param p Current value of p (N*n x 1)
/// @param s Current value of s from s_table (N*n x N*n)
/// @param sys System matrices containing weighted_Q, BR_sum
/// @param xd Desired state at current time (N*n x 1)
/// @return Rate of change of p (N*n x 1)
VectorXd p_dot(const VectorXd& p, const MatrixXd& s, const SystemParams& sys, const VectorXd& xd) {
    return sys.weighted_Q * xd + s * sys.BR_sum * p;
}

// ============================================================
// RK4 backward steps
// ============================================================

/// @brief Single RK4 backward step for s(t)
/// Takes s at step k+1, returns s at step k
/// @param s Current s value
/// @param dt Timestep size
/// @param sys System matrices
/// @return s one step earlier in time
MatrixXd rk4_step_s(const MatrixXd& s, double dt, const SystemParams& sys) {
  MatrixXd k1 = s_dot(s, sys);
  MatrixXd s2 = s - (0.5 * dt * k1);
  MatrixXd k2 = s_dot(s2, sys);
  MatrixXd s3 = s - (0.5 * dt * k2);
  MatrixXd k3 = s_dot(s3, sys);
  MatrixXd s4 = s - (dt * k3);
  MatrixXd k4 = s_dot(s4,sys);
  return s - dt/6 * (k1 + 2*k2 + 2*k3+ k4);
}

/// @brief Single RK4 backward step for p(t)
/// Takes p at step k+1, returns p at step k
/// Needs s at the same step and desired state at that time
/// @param p Current p value
/// @param s Current s value from s_table
/// @param dt Timestep size
/// @param sys System matrices
/// @param xd Desired state at current time
/// @return p one step earlier in time
VectorXd rk4_step_p(const VectorXd& p, const MatrixXd& s, double dt, const SystemParams& sys, const VectorXd& xd) {

  VectorXd k1 = p_dot(p, s, sys, xd);
  VectorXd p2 = p - (0.5 * dt * k1);
  VectorXd k2 = p_dot(p2, s, sys, xd);
  VectorXd p3 = p - (0.5 * dt * k2);
  VectorXd k3 = p_dot(p3, s, sys, xd);
  VectorXd p4 = p - (dt * k3);
  VectorXd k4 = p_dot(p4, s, sys, xd);
  return p - dt/6 * (k1 + 2*k2 + 2*k3+ k4);
}

// ============================================================
// Main solver
// ============================================================

SolverResult solver_precompute(const SystemParams& sys, const Config& cfg) {
    int K = sys.num_steps;
    SolverResult result;
    result.num_steps = K;
    result.dt = cfg.dt;
    result.s_table.resize(K + 1);
    result.p_table.resize(K + 1);


    // --- Terminal conditions ---
    VectorXd xd_T = get_desired_state(cfg, cfg.t_total);

    result.p_table[K] = -sys.weighted_Q * xd_T;
    result.s_table[K] = sys.weighted_Q;
    //
    // --- Backward integration of s ---
    for (int k = K - 1; k >= 0; k--) {
        double t = (k + 1) * cfg.dt;
        VectorXd xd = get_desired_state(cfg, t);

        result.s_table[k] = rk4_step_s(result.s_table[k + 1], cfg.dt, sys);
        result.p_table[k] = rk4_step_p(result.p_table[k + 1],
                                         result.s_table[k + 1],
                                         cfg.dt, sys, xd);
    }

    return result;
}

// ============================================================
// File IO
// ============================================================

void solver_save(const SolverResult& result, const std::string& filename) {
    // TODO: write s_table and p_table to binary file
}

SolverResult solver_load(const std::string& filename) {
    // TODO: read s_table and p_table from binary file
    SolverResult result;
    return result;
}
