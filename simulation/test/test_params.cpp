// test_params.cpp
// Tests params_build() using the paper's configuration (Tables I-IV)
// Checks matrix dimensions, known values, and structural properties

#include "params.h"
#include <iostream>
#include <cassert>
#include <cmath>

// Helper: check matrix dimensions
void check_dims(const MatrixXd& M, int rows, int cols, const std::string& name) {
    if (M.rows() == rows && M.cols() == cols) {
        std::cout << "  PASS: " << name << " is " << rows << "x" << cols << std::endl;
    } else {
        std::cout << "  FAIL: " << name << " is " << M.rows() << "x" << M.cols()
                  << " (expected " << rows << "x" << cols << ")" << std::endl;
    }
}

// Helper: check if value matches expected
void check_val(double actual, double expected, const std::string& name, double tol = 1e-10) {
    if (std::abs(actual - expected) < tol) {
        std::cout << "  PASS: " << name << " = " << actual << std::endl;
    } else {
        std::cout << "  FAIL: " << name << " = " << actual
                  << " (expected " << expected << ")" << std::endl;
    }
}

// Helper: check matrix is symmetric
void check_symmetric(const MatrixXd& M, const std::string& name, double tol = 1e-10) {
    double diff = (M - M.transpose()).norm();
    if (diff < tol) {
        std::cout << "  PASS: " << name << " is symmetric" << std::endl;
    } else {
        std::cout << "  FAIL: " << name << " asymmetry norm = " << diff << std::endl;
    }
}

// Build the paper's configuration (Tables I-IV, N=5)
Config make_paper_config() {
    Config cfg;

    cfg.num_drones = 5;
    cfg.leader_id = 0;

    cfg.mass = 0.027;
    cfg.g = 9.81;
    cfg.a_theta = 10.0;   // assumed — not given in paper
    cfg.a_phi   = 10.0;   // assumed — not given in paper
    cfg.a_r     = 10.0;   // assumed — not given in paper

    // Table I: formation offsets (dx, dy, dz)
    cfg.formation_offsets = MatrixXd(5, 3);
    cfg.formation_offsets <<  1,  4,  3,
                             -1, -2,  1,
                              3,  1, -2,
                             -3, -3, -1,
                             -4, -4, -4;

    // Fig 1: communication topology (tree)
    cfg.edge_list = Eigen::MatrixXi(4, 2);
    cfg.edge_list << 1, 0,
                     2, 0,
                     3, 1,
                     4, 2;

    // Table II: all edge weights = 5
    cfg.edge_weights = VectorXd::Constant(4, 5.0);

    // Equal team importance
    cfg.alpha = VectorXd(5);
    cfg.alpha << 0.2, 0.2, 0.2, 0.2, 0.2;

    // Unoptimised: all R_i = 1
    cfg.R_diag = VectorXd::Constant(5, 1.0);

    // Leader trajectory tracking weight
    cfg.q_leader = 5.0;

    // Table III: waypoints
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
    cfg.t_total = 120.0;

    return cfg;
}

int main() {
    Config cfg = make_paper_config();
    int N = cfg.num_drones;
    int n = STATES_PER_DRONE;
    int m = INPUTS_PER_DRONE;

    std::cout << "========================================" << std::endl;
    std::cout << "Building system params..." << std::endl;
    std::cout << "========================================" << std::endl;
    SystemParams sys = params_build(cfg);

    // ---- Basic dimensions ----
    std::cout << "\n--- Dimension checks ---" << std::endl;
    check_dims(sys.A, N * n, N * n, "A (full system)");
    check_dims(sys.D, N, 4, "D (incidence)");
    check_dims(sys.weighted_Q, N * n, N * n, "weighted_Q");
    check_dims(sys.BR_sum, N * n, N * n, "BR_sum");

    for (int i = 0; i < N; i++) {
        check_dims(sys.B[i], N * n, m, "B[" + std::to_string(i) + "]");
        check_dims(sys.Q[i], N * n, N * n, "Q[" + std::to_string(i) + "]");
        check_dims(sys.R[i], m, m, "R[" + std::to_string(i) + "]");
    }

    // ---- A matrix structure ----
    std::cout << "\n--- A matrix checks ---" << std::endl;

    // A should be block diagonal: check that off-diagonal blocks are zero
    double off_diag_norm = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j) {
                off_diag_norm += sys.A.block(i * n, j * n, n, n).norm();
            }
        }
    }
    check_val(off_diag_norm, 0.0, "A off-diagonal blocks norm");

    // Check specific A_i values: A(0,1) should be 1 (x_dot = velocity)
    check_val(sys.A(0, 1), 1.0, "A(0,1) = 1 (x kinematics)");
    // A(1,2) should be -g
    check_val(sys.A(1, 2), -cfg.g, "A(1,2) = -g (x acceleration from pitch)");
    // A(2,2) should be -a_theta
    check_val(sys.A(2, 2), -cfg.a_theta, "A(2,2) = -a_theta (pitch decay)");

    // Drone 2 (index 1) should have same A_i block at rows 10-19
    check_val(sys.A(10, 11), 1.0, "A(10,11) = 1 (drone 2 x kinematics)");
    check_val(sys.A(11, 12), -cfg.g, "A(11,12) = -g (drone 2 x accel)");

    // ---- B matrix structure ----
    std::cout << "\n--- B matrix checks ---" << std::endl;

    // B[0] should have non-zero entries only in rows 0-9 (drone 0's states)
    double B0_wrong_rows = sys.B[0].block(n, 0, (N - 1) * n, m).norm();
    check_val(B0_wrong_rows, 0.0, "B[0] rows 10-49 all zero");

    // B[0](2,0) should be a_theta (pitch command affects pitch state)
    check_val(sys.B[0](2, 0), cfg.a_theta, "B[0](2,0) = a_theta");

    // B[2] should have non-zero entries only in rows 20-29 (drone 2's states)
    double B2_top = sys.B[2].block(0, 0, 2 * n, m).norm();
    double B2_bot = sys.B[2].block(3 * n, 0, 2 * n, m).norm();
    check_val(B2_top, 0.0, "B[2] rows 0-19 all zero");
    check_val(B2_bot, 0.0, "B[2] rows 30-49 all zero");

    // ---- D matrix checks ----
    std::cout << "\n--- D matrix checks ---" << std::endl;

    // Paper's D matrix from Fig 1
    // Edge 0: drone 1->0, so D(1,0)=-1, D(0,0)=+1
    check_val(sys.D(0, 0),  1.0, "D(0,0) = +1 (drone 0 receives edge 0)");
    check_val(sys.D(1, 0), -1.0, "D(1,0) = -1 (drone 1 sends edge 0)");

    // Edge 3: drone 4->2, so D(4,3)=-1, D(2,3)=+1
    check_val(sys.D(4, 3), -1.0, "D(4,3) = -1 (drone 4 sends edge 3)");
    check_val(sys.D(2, 3),  1.0, "D(2,3) = +1 (drone 2 receives edge 3)");

    // Each column should have exactly one +1 and one -1
    for (int e = 0; e < 4; e++) {
        double col_sum = sys.D.col(e).sum();
        check_val(col_sum, 0.0, "D column " + std::to_string(e) + " sum = 0");
    }

    // ---- Q matrix properties ----
    std::cout << "\n--- Q matrix checks ---" << std::endl;

    // All Q_i should be symmetric
    for (int i = 0; i < N; i++) {
        check_symmetric(sys.Q[i], "Q[" + std::to_string(i) + "]");
    }

    // Q_i should be positive semidefinite (all eigenvalues >= 0)
    for (int i = 0; i < N; i++) {
        Eigen::SelfAdjointEigenSolver<MatrixXd> solver(sys.Q[i]);
        double min_eigenval = solver.eigenvalues().minCoeff();
        if (min_eigenval >= -1e-10) {
            std::cout << "  PASS: Q[" << i << "] is positive semidefinite (min eig = "
                      << min_eigenval << ")" << std::endl;
        } else {
            std::cout << "  FAIL: Q[" << i << "] min eigenvalue = "
                      << min_eigenval << std::endl;
        }
    }

    // Leader Q (index 0) should have larger diagonal than followers
    // because of the trajectory tracking term
    double leader_diag = sys.Q[0].diagonal().sum();
    double follower_diag = sys.Q[1].diagonal().sum();
    if (leader_diag > follower_diag) {
        std::cout << "  PASS: Leader Q diagonal (" << leader_diag
                  << ") > follower Q diagonal (" << follower_diag << ")" << std::endl;
    } else {
        std::cout << "  FAIL: Leader Q diagonal not larger than follower" << std::endl;
    }

    // ---- R matrix checks ----
    std::cout << "\n--- R matrix checks ---" << std::endl;
    check_val(sys.R[0](0, 0), 1.0, "R[0](0,0) = 1.0 (unoptimised)");
    check_val(sys.R[0](1, 1), 1.0, "R[0](1,1) = 1.0 (unoptimised)");
    // R should be diagonal
    double R_offdiag = sys.R[0](0, 1);
    check_val(R_offdiag, 0.0, "R[0] off-diagonal = 0");

    // ---- Precomputed constants checks ----
    std::cout << "\n--- Precomputed constants checks ---" << std::endl;
    check_symmetric(sys.weighted_Q, "weighted_Q");
    check_symmetric(sys.BR_sum, "BR_sum");

    // weighted_Q should be positive semidefinite
    Eigen::SelfAdjointEigenSolver<MatrixXd> wq_solver(sys.weighted_Q);
    double wq_min_eig = wq_solver.eigenvalues().minCoeff();
    if (wq_min_eig >= -1e-10) {
        std::cout << "  PASS: weighted_Q positive semidefinite (min eig = "
                  << wq_min_eig << ")" << std::endl;
    } else {
        std::cout << "  FAIL: weighted_Q min eigenvalue = " << wq_min_eig << std::endl;
    }

    // BR_sum should be positive semidefinite
    Eigen::SelfAdjointEigenSolver<MatrixXd> br_solver(sys.BR_sum);
    double br_min_eig = br_solver.eigenvalues().minCoeff();
    if (br_min_eig >= -1e-10) {
        std::cout << "  PASS: BR_sum positive semidefinite (min eig = "
                  << br_min_eig << ")" << std::endl;
    } else {
        std::cout << "  FAIL: BR_sum min eigenvalue = " << br_min_eig << std::endl;
    }

    // ---- Desired state checks ----
    std::cout << "\n--- Desired state checks ---" << std::endl;

    VectorXd xd = get_desired_state(cfg, 10.0);  // t = 10 seconds
    check_dims(xd, N * n, 1, "desired_state at t=10");

    // At t=10, straight line trajectory: x = 0.2 * 10 = 2.0
    // Drone 0 offset_x = 1, so desired x = 2.0 + 1.0 = 3.0
    check_val(xd(0), 3.0, "xd(0) = traj_x + offset_x for drone 0");

    // Drone 0 desired x-velocity should be mean_velocity = 0.2
    check_val(xd(1), 0.2, "xd(1) = traj_vx for drone 0");

    // Drone 0 desired pitch should be 0
    check_val(xd(2), 0.0, "xd(2) = 0 (desired level flight)");

    // Drone 1 (index 1): offset_x = -1, so desired x = 2.0 + (-1) = 1.0
    check_val(xd(10), 1.0, "xd(10) = traj_x + offset_x for drone 1");

    // All drones should have same velocity
    check_val(xd(1), xd(11), "drone 0 and drone 1 same x-velocity");

    // ---- Print summary info ----
    std::cout << "\n--- Summary ---" << std::endl;
    std::cout << "N = " << sys.N << " drones" << std::endl;
    std::cout << "Total states = " << sys.total_states << std::endl;
    std::cout << "Total inputs = " << sys.total_inputs << std::endl;
    std::cout << "Num steps = " << sys.num_steps << std::endl;
    std::cout << "A matrix non-zeros: " << (sys.A.array() != 0).count()
              << " / " << sys.A.size() << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Tests complete." << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
