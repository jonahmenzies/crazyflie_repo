// test_solver.cpp
#include "solver.h"
#include <iostream>
#include <cmath>

void check_dims_m(const MatrixXd& M, int rows, int cols, const std::string& name) {
    if (M.rows() == rows && M.cols() == cols) {
        std::cout << "  PASS: " << name << " is " << rows << "x" << cols << std::endl;
    } else {
        std::cout << "  FAIL: " << name << " is " << M.rows() << "x" << M.cols()
                  << " (expected " << rows << "x" << cols << ")" << std::endl;
    }
}

void check_val(double actual, double expected, const std::string& name, double tol = 1e-10) {
    if (std::abs(actual - expected) < tol) {
        std::cout << "  PASS: " << name << " = " << actual << std::endl;
    } else {
        std::cout << "  FAIL: " << name << " = " << actual
                  << " (expected " << expected << ")" << std::endl;
    }
}

void check_symmetric(const MatrixXd& M, const std::string& name, double tol = 1e-6) {
    double diff = (M - M.transpose()).norm();
    if (diff < tol) {
        std::cout << "  PASS: " << name << " is symmetric (diff = " << diff << ")" << std::endl;
    } else {
        std::cout << "  FAIL: " << name << " asymmetry norm = " << diff << std::endl;
    }
}

Config make_test_config() {
    Config cfg;
    cfg.num_drones = 5;
    cfg.leader_id = 0;
    cfg.mass = 0.027;
    cfg.g = 9.81;
    cfg.a_theta = 10.0;
    cfg.a_phi   = 10.0;
    cfg.a_r     = 10.0;

    cfg.formation_offsets = MatrixXd(5, 3);
    cfg.formation_offsets <<  1,  4,  3,
                             -1, -2,  1,
                              3,  1, -2,
                             -3, -3, -1,
                             -4, -4, -4;

    cfg.edge_list = Eigen::MatrixXi(4, 2);
    cfg.edge_list << 1, 0, 2, 0, 3, 1, 4, 2;

    cfg.edge_weights = VectorXd::Constant(4, 5.0);
    cfg.alpha = VectorXd::Constant(5, 0.2);
    cfg.R_diag = VectorXd::Constant(5, 1000.0);
    cfg.q_leader = 5.0;

    cfg.waypoints = MatrixXd(10, 3);
    cfg.waypoints <<  0.0, 0.0, 1.0,
                      1.0, 0.0, 1.0,
                      1.0, 1.0, 2.0,
                      2.0, 1.0, 1.0,
                      2.0, 0.0, 1.0,
                      3.5, 0.5, 1.0,
                      3.5, 1.5, 3.5,
                      2.5, 2.5, 3.5,
                      3.0, 3.5, 2.5,
                      2.0, 4.0, 1.5;

    cfg.mean_velocity = 0.2;
    cfg.min_turn_radius = 0.15;
    cfg.dt = 0.01;
    cfg.t_total = 2.0;

    return cfg;
}

int main() {
    Config cfg = make_test_config();
    int N = cfg.num_drones;
    int n = STATES_PER_DRONE;
    int total = N * n;
    int K = static_cast<int>(cfg.t_total / cfg.dt);

    std::cout << "========================================" << std::endl;
    std::cout << "Building system params..." << std::endl;
    SystemParams sys = params_build(cfg);
    std::cout << "BR_sum norm: " << sys.BR_sum.norm() << std::endl;
    std::cout << "weighted_Q norm: " << sys.weighted_Q.norm() << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Running solver (K=" << K << " steps)..." << std::endl;
    std::cout << "========================================" << std::endl;
    SolverResult sol = solver_precompute(sys, cfg);

    std::cout << "\n--- Table size checks ---" << std::endl;
    check_val(sol.s_table.size(), K + 1, "s_table length");
    check_val(sol.p_table.size(), K + 1, "p_table length");

    std::cout << "\n--- Terminal condition checks ---" << std::endl;
    double s_terminal_diff = (sol.s_table[K] - sys.weighted_Q).norm();
    check_val(s_terminal_diff, 0.0, "s(t_T) matches weighted_Q");

    VectorXd xd_T = get_desired_state(cfg, cfg.t_total);
    VectorXd expected_p_T = -sys.weighted_Q * xd_T;
    double p_terminal_diff = (sol.p_table[K] - expected_p_T).norm();
    check_val(p_terminal_diff, 0.0, "p(t_T) matches -weighted_Q * xd_T");

    std::cout << "\n--- Symmetry checks ---" << std::endl;
    check_symmetric(sol.s_table[0], "s_table[0]");
    check_symmetric(sol.s_table[K / 2], "s_table[K/2]");
    check_symmetric(sol.s_table[K], "s_table[K]");

    std::cout << "\n--- Positive semidefinite checks ---" << std::endl;
    for (int idx : {0, K / 4, K / 2, 3 * K / 4, K}) {
        Eigen::SelfAdjointEigenSolver<MatrixXd> solver(sol.s_table[idx]);
        double min_eig = solver.eigenvalues().minCoeff();
        std::string name = "s_table[" + std::to_string(idx) + "]";
        if (min_eig >= -1e-6) {
            std::cout << "  PASS: " << name << " min eig = " << min_eig << std::endl;
        } else {
            std::cout << "  FAIL: " << name << " min eig = " << min_eig << std::endl;
        }
    }

    std::cout << "\n--- Evolution checks ---" << std::endl;
    double s_diff = (sol.s_table[0] - sol.s_table[K]).norm();
    if (s_diff > 1e-6) {
        std::cout << "  PASS: s evolves over time (diff = " << s_diff << ")" << std::endl;
    } else {
        std::cout << "  FAIL: s doesn't change" << std::endl;
    }

    double p_diff = (sol.p_table[0] - sol.p_table[K]).norm();
    if (p_diff > 1e-6) {
        std::cout << "  PASS: p evolves over time (diff = " << p_diff << ")" << std::endl;
    } else {
        std::cout << "  FAIL: p doesn't change" << std::endl;
    }

    std::cout << "\n--- Stability checks ---" << std::endl;
    bool all_finite = true;
    for (int k = 0; k <= K; k++) {
        if (!sol.s_table[k].allFinite() || !sol.p_table[k].allFinite()) {
            std::cout << "  FAIL: NaN/Inf at step " << k << std::endl;
            all_finite = false;
            break;
        }
    }
    if (all_finite) std::cout << "  PASS: all values finite" << std::endl;

    std::cout << "\n--- Sample norms ---" << std::endl;
    std::cout << "  s_table[0]:   " << sol.s_table[0].norm() << std::endl;
    std::cout << "  s_table[K/2]: " << sol.s_table[K / 2].norm() << std::endl;
    std::cout << "  s_table[K]:   " << sol.s_table[K].norm() << std::endl;
    std::cout << "  p_table[0]:   " << sol.p_table[0].norm() << std::endl;
    std::cout << "  p_table[K/2]: " << sol.p_table[K / 2].norm() << std::endl;
    std::cout << "  p_table[K]:   " << sol.p_table[K].norm() << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Tests complete." << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
