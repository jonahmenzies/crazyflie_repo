// params.cpp
#include "params.h"
#include <iostream>
#include <cmath>

// ============================================================
// Forward declarations
// ============================================================

void get_trajectory_at_t(const Config& cfg, double t,
                         double& x, double& y, double& z,
                         double& vx, double& vy, double& vz,
                         double& yaw);

// ============================================================
// Utility
// ============================================================

/// @brief General Kronecker product: A ⊗ B
/// Replaces each element A(i,j) with A(i,j) * B in the output.
/// Output size: (A.rows * B.rows) x (A.cols * B.cols)
/// @param A Left matrix
/// @param B Right matrix
/// @return A ⊗ B
MatrixXd kronecker(const MatrixXd& A, const MatrixXd& B) {
    MatrixXd result = MatrixXd::Zero(A.rows() * B.rows(), A.cols() * B.cols());
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            result.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
        }
    }
    return result;
}

// ============================================================
// Single-drone matrices
// ============================================================

/// @brief Builds the single-drone dynamics matrix A_i (10x10)
/// Encodes linearised physics at hover: how the drone state evolves naturally.
/// Decoupled into 4 subsystems: x-axis (rows 0-2), y-axis (rows 3-5),
/// z-axis (rows 6-7), yaw (rows 8-9).
/// @param cfg Config containing g, a_theta, a_phi, a_r
/// @return A_i matrix (10x10)
MatrixXd build_Ai(const Config& cfg) {
    MatrixXd A = MatrixXd::Zero(10, 10);
//             x   ẋ              θ   y   ẏ              ϕ   z   ż   ψ              r
    A <<       0,  1,             0,  0,  0,             0,  0,  0,  0,             0,
               0,  0,        -cfg.g,  0,  0,             0,  0,  0,  0,             0,
               0,  0,  -cfg.a_theta,  0,  0,             0,  0,  0,  0,             0,
               0,  0,             0,  0,  1,             0,  0,  0,  0,             0,
               0,  0,             0,  0,  0,        -cfg.g,  0,  0,  0,             0,
               0,  0,             0,  0,  0,    -cfg.a_phi,  0,  0,  0,             0,
               0,  0,             0,  0,  0,             0,  0,  1,  0,             0,
               0,  0,             0,  0,  0,             0,  0,  0,  0,             0,
               0,  0,             0,  0,  0,             0,  0,  0,  0,             1,
               0,  0,             0,  0,  0,             0,  0,  0,  0,      -cfg.a_r;
    return A;
}

/// @brief Builds the single-drone input matrix b_i (10x4)
/// Maps control commands to state derivatives:
///   col 0: pitch command (theta_d) → x-axis motion
///   col 1: roll command (phi_d) → y-axis motion
///   col 2: thrust command (deltaF_d) → z-axis motion
///   col 3: yaw rate command (r_d) → heading
/// @param cfg Config containing a_theta, a_phi, a_r, mass
/// @return b_i matrix (10x4)
MatrixXd build_bi(const Config& cfg) {
    MatrixXd b = MatrixXd::Zero(10, 4);
//                θd             ϕd              δFd             rd
    b <<          0,              0,               0,              0,   // x
                  0,              0,               0,              0,   // ẋ
        cfg.a_theta,              0,               0,              0,   // θ
                  0,              0,               0,              0,   // y
                  0,              0,               0,              0,   // ẏ
                  0,    cfg.a_phi,                 0,              0,   // ϕ
                  0,              0,               0,              0,   // z
                  0,              0,  -1.0/cfg.mass,              0,   // ż
                  0,              0,               0,              0,   // ψ
                  0,              0,               0,       -cfg.a_r;   // r
    return b;
}

// ============================================================
// Full system matrices
// ============================================================

/// @brief Builds the full system dynamics matrix A (N*n x N*n)
/// Block-diagonal: each drone's A_i placed along the diagonal.
/// Equivalent to I_N ⊗ A_i (Kronecker product)
/// @param cfg Config containing num_drones and physical parameters
/// @return A matrix (N*n x N*n)
MatrixXd build_A_full(const Config& cfg) {
    MatrixXd I_N = MatrixXd::Identity(cfg.num_drones, cfg.num_drones);
    MatrixXd Ai = build_Ai(cfg);
    return kronecker(I_N, Ai);
}

/// @brief Builds the full input matrix B_i for drone i (N*n x m)
/// Mostly zeros — the single-drone b_i is placed at drone i's rows.
/// When multiplied by u_i, only drone i's states are affected.
/// @param cfg Config containing num_drones and physical parameters
/// @param i Drone index (0-indexed)
/// @return B_i matrix (N*n x m)
MatrixXd build_Bi(const Config& cfg, int i) {
    MatrixXd bi = build_bi(cfg);
    int N = cfg.num_drones;
    int n = STATES_PER_DRONE;
    int m = INPUTS_PER_DRONE;
    MatrixXd Bi = MatrixXd::Zero(N * n, m);
    Bi.block(i * n, 0, n, m) = bi;
    return Bi;
}

// ============================================================
// Communication topology
// ============================================================

/// @brief Builds the incidence matrix D defining the communication topology
/// D is (num_drones x num_edges). Each column represents one edge:
///   -1 at the sender's row, +1 at the receiver's row, 0 elsewhere.
/// @param cfg Config containing edge_list (num_edges x 2) and num_drones
/// @return D matrix (num_drones x num_edges)
MatrixXd build_D(const Config& cfg) {
    int num_edges = cfg.edge_list.rows();
    MatrixXd D = MatrixXd::Zero(cfg.num_drones, num_edges);
    for (int e = 0; e < num_edges; e++) {
        int sender   = cfg.edge_list(e, 0);
        int receiver = cfg.edge_list(e, 1);
        D(sender, e)   = -1.0;
        D(receiver, e) =  1.0;
    }
    return D;
}

/// @brief Builds edge weight matrix W_i for drone i (num_edges x num_edges)
/// Diagonal matrix: non-zero on edges drone i is connected to, zero elsewhere.
/// Built from edge_list and edge_weights — no extra config needed.
/// @param cfg Config containing edge_list and edge_weights
/// @param i Drone index (0-indexed)
/// @return W_i diagonal matrix (num_edges x num_edges)
MatrixXd build_Wi(const Config& cfg, int i) {
    int num_edges = cfg.edge_list.rows();
    MatrixXd Wi = MatrixXd::Zero(num_edges, num_edges);
    for (int e = 0; e < num_edges; e++) {
        int sender   = cfg.edge_list(e, 0);
        int receiver = cfg.edge_list(e, 1);
        if (sender == i || receiver == i) {
            Wi(e, e) = cfg.edge_weights(e);
        }
    }
    return Wi;
}

// ============================================================
// Cost matrices
// ============================================================

/// @brief Builds state cost matrix Q_i for drone i (N*n x N*n)
/// Q_i = D_hat * W_hat_i * D_hat^T where D_hat = D ⊗ I_n, W_hat_i = W_i ⊗ I_n
/// Penalises formation deviation between drone i and its neighbours.
/// @param cfg Config for building W_i
/// @param i Drone index (0-indexed)
/// @param D Incidence matrix from build_D
/// @return Q_i matrix (N*n x N*n)
MatrixXd build_Qi(const Config& cfg, int i, const MatrixXd& D) {
    MatrixXd In = MatrixXd::Identity(STATES_PER_DRONE, STATES_PER_DRONE);
    MatrixXd D_hat = kronecker(D, In);
    MatrixXd Wi = build_Wi(cfg, i);
    MatrixXd W_hat_i = kronecker(Wi, In);
    return D_hat * W_hat_i * D_hat.transpose();
}

/// @brief Adds trajectory tracking term to the leader's Q_i (eq 6)
/// Adds q * I_n at the leader's block on the diagonal
/// @param cfg Config containing leader_id and q_leader
/// @param Qi_base The leader's base Q_i from build_Qi
/// @return Modified Q_i with trajectory tracking penalty
MatrixXd build_Qi_leader(const Config& cfg, const MatrixXd& Qi_base) {
    MatrixXd Q = Qi_base;
    int n = STATES_PER_DRONE;
    int idx = cfg.leader_id * n;
    Q.block(idx, idx, n, n) += cfg.q_leader * MatrixXd::Identity(n, n);
    return Q;
}

/// @brief Builds the control effort cost matrix R_i for drone i (4x4)
/// Diagonal matrix scaling how expensive control inputs are.
/// Higher R_i = gentler control, more formation drift.
/// Lower R_i = aggressive control, tighter formation.
/// Paper uses R_i = 1 for unoptimised, [5600, 1005, 1050, 1050, 2920] for optimised.
/// @param cfg Config containing R_diag (per-drone scalar weight)
/// @param i Drone index (0-indexed)
/// @return R_i matrix (4x4)
MatrixXd build_Ri(const Config& cfg, int i) {
    return cfg.R_diag(i) * MatrixXd::Identity(INPUTS_PER_DRONE, INPUTS_PER_DRONE);
}

// ============================================================
// Precomputed constants
// ============================================================

/// @brief Computes the weighted sum of formation cost matrices: Σ(alpha_i * Q_i)
/// Appears in equations 9b and 9c. Constant across all timesteps so computed once.
/// @param cfg Config containing alpha (team payoff weights)
/// @param Q Vector of N cost matrices Q_i, each (N*n x N*n)
/// @return Weighted sum matrix (N*n x N*n)
MatrixXd build_weighted_Q(const Config& cfg, const std::vector<MatrixXd>& Q) {
    MatrixXd result = MatrixXd::Zero(Q[0].rows(), Q[0].cols());
    for (int i = 0; i < cfg.num_drones; i++) {
        result += cfg.alpha(i) * Q[i];
    }
    return result;
}

/// @brief Computes the weighted control term: Σ((1/alpha_i) * B_i * R_i^-1 * B_i^T)
/// Appears in equations 9b and 9c. Constant across all timesteps so computed once.
/// @param cfg Config containing alpha (team payoff weights)
/// @param B Vector of N input matrices B_i, each (N*n x m)
/// @param R Vector of N control cost matrices R_i, each (m x m)
/// @return Weighted sum matrix (N*n x N*n)
MatrixXd build_BR_sum(const Config& cfg, const std::vector<MatrixXd>& B, const std::vector<MatrixXd>& R) {
    MatrixXd result = MatrixXd::Zero(B[0].rows(), B[0].rows());
    for (int i = 0; i < cfg.num_drones; i++) {
        result += (1.0 / cfg.alpha(i)) * B[i] * R[i].inverse() * B[i].transpose();
    }
    return result;
}

// ============================================================
// Trajectory
// ============================================================

/// @brief Temporary path manager — straight line along x-axis
/// Replace with full waypoint interpolation + corner smoothing for paper replication.
/// @param cfg Config containing mean_velocity
/// @param t Current time (seconds)
/// @param[out] x, y, z Trajectory position at time t (metres)
/// @param[out] vx, vy, vz Trajectory velocity at time t (m/s)
/// @param[out] yaw Trajectory heading at time t (rad)
void get_trajectory_at_t(const Config& cfg, double t,
                         double& x, double& y, double& z,
                         double& vx, double& vy, double& vz,
                         double& yaw) {
    // Straight line along x-axis at constant velocity
    x  = cfg.mean_velocity * t;
    y  = 0.0;
    z  = 1.0;
    vx = cfg.mean_velocity;
    vy = 0.0;
    vz = 0.0;
    yaw = 0.0;
}

/// @brief Computes desired state vector for all drones at time t
/// Position states get per-drone formation offsets (relative consensus).
/// Velocity and attitude states are identical across drones (absolute consensus).
/// Desired pitch, roll, and yaw rate are zero (level flight, steady heading).
/// @param cfg Config containing formation_offsets, num_drones, mean_velocity
/// @param t Current time (seconds)
/// @return Desired state vector (N*n x 1)
VectorXd get_desired_state(const Config& cfg, double t) {
    double traj_x, traj_y, traj_z;
    double traj_vx, traj_vy, traj_vz;
    double traj_yaw;
    get_trajectory_at_t(cfg, t, traj_x, traj_y, traj_z,
                        traj_vx, traj_vy, traj_vz, traj_yaw);

    VectorXd desired_state(cfg.num_drones * STATES_PER_DRONE);

    for (int i = 0; i < cfg.num_drones; i++) {
        double offset_x = cfg.formation_offsets(i, 0);
        double offset_y = cfg.formation_offsets(i, 1);
        double offset_z = cfg.formation_offsets(i, 2);

        VectorXd state(STATES_PER_DRONE);
        state << traj_x + offset_x,   // x  — trajectory position + drone i's offset
                 traj_vx,              // ẋ  — trajectory velocity (same for all drones)
                 0.0,                  // θ  — desired is level flight
                 traj_y + offset_y,    // y  — trajectory position + drone i's offset
                 traj_vy,              // ẏ  — trajectory velocity (same for all drones)
                 0.0,                  // ϕ  — desired is level flight
                 traj_z + offset_z,    // z  — trajectory position + drone i's offset
                 traj_vz,              // ż  — trajectory velocity (same for all drones)
                 traj_yaw,             // ψ  — trajectory heading (same for all drones)
                 0.0;                  // r  — desired is steady heading

        desired_state.segment(i * STATES_PER_DRONE, STATES_PER_DRONE) = state;
    }

    return desired_state;
}

// ============================================================
// Main builder — ties everything together
// ============================================================

/// @brief Constructs all system matrices from configuration
/// Builds dynamics (A, B), topology (D), cost matrices (Q, R),
/// and precomputes constant terms for the solver.
/// @param cfg Complete mission configuration
/// @return SystemParams containing all matrices needed by solver and controller
SystemParams params_build(const Config& cfg) {
    SystemParams sys;
    sys.N = cfg.num_drones;
    sys.n = STATES_PER_DRONE;
    sys.m = INPUTS_PER_DRONE;
    sys.total_states = sys.N * sys.n;
    sys.total_inputs = sys.N * sys.m;
    sys.num_steps = static_cast<int>(cfg.t_total / cfg.dt);

    // Build system dynamics
    sys.A = build_A_full(cfg);

    // Build input matrices for each drone
    sys.B.resize(sys.N);
    for (int i = 0; i < sys.N; i++) {
        sys.B[i] = build_Bi(cfg, i);
    }

    // Build communication topology
    sys.D = build_D(cfg);

    // Build cost matrices for each drone
    sys.Q.resize(sys.N);
    sys.Q_T.resize(sys.N);
    sys.R.resize(sys.N);
    for (int i = 0; i < sys.N; i++) {
        sys.Q[i] = build_Qi(cfg, i, sys.D);
        if (i == cfg.leader_id) {
            sys.Q[i] = build_Qi_leader(cfg, sys.Q[i]);
        }
        sys.Q_T[i] = sys.Q[i];  // terminal cost same as running cost
        sys.R[i] = build_Ri(cfg, i);
    }

    // Precompute constant terms for solver
    sys.weighted_Q = build_weighted_Q(cfg, sys.Q);
    sys.BR_sum = build_BR_sum(cfg, sys.B, sys.R);

    return sys;
}
