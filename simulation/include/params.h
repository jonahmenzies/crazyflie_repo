#pragma once
#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// ============================================================
// Constants
// ============================================================

constexpr int    STATES_PER_DRONE  = 10;   // (x, xdot, theta, y, ydot, phi, z, zdot, psi, r)
constexpr int    INPUTS_PER_DRONE  = 4;    // (theta_d, phi_d, deltaF_d, r_d)
constexpr double MAX_PITCH         = M_PI / 4.0;   // rad — inclination protection limit
constexpr double MAX_ROLL          = M_PI / 4.0;   // rad — inclination protection limit
constexpr double MAX_YAW_RATE      = 2.0 * M_PI;   // rad/s — yaw rate limit
constexpr double CRAZYFLIE_MASS    = 0.027;         // kg
constexpr double GRAVITY           = 9.81;          // m/s^2

// ============================================================
// Control mode
// ============================================================

enum class ControlMode {
    CLOSED_LOOP,       // x(t) from measurements every tick
    OPEN_LOOP,         // x(t) estimated from initial conditions and model
    SEMI_OPEN_LOOP     // x(t) estimated, reset from measurements every Ts seconds
};

// ============================================================
// Configuration — what you define for a mission
// ============================================================

struct Config {
    int num_drones;                // N — total number of drones in the team
    int leader_id;                 // 0-indexed — which drone tracks the trajectory

    // System ID parameters — first-order attitude response constants
    // These characterise how fast the inner-loop controller tracks commands
    // Obtained via step response testing on the Crazyflie
    double mass;                   // kg — total drone mass
    double g;                      // m/s^2 — gravitational acceleration
    double a_theta;                // 1/s — pitch response rate
    double a_phi;                  // 1/s — roll response rate
    double a_r;                    // 1/s — yaw rate response rate

    // Formation geometry
    // N x 3 matrix — each row is (dx, dy, dz) offset in metres
    // from the central trajectory for drone i
    MatrixXd formation_offsets;

    // Communication topology
    // edge_list: num_edges x 2 matrix, each row is (sender_id, receiver_id), 0-indexed
    // edge_weights: num_edges x 1 vector, weight per edge (paper uses 5.0 for all)
    // Direction doesn't significantly affect performance (paper remark Section III)
    Eigen::MatrixXi edge_list;
    VectorXd edge_weights;

    // Game theory weights
    // alpha: N x 1 vector, team payoff weights — must sum to 1.0
    //        higher alpha_i = team prioritises drone i's cost more
    // R_diag: N x 1 vector, control input weight per drone
    //         higher R_i = gentler control, worse tracking
    //         lower R_i  = aggressive control, better tracking
    // q_leader: scalar weight for leader's trajectory tracking term (eq 6)
    VectorXd alpha;
    VectorXd R_diag;
    double q_leader;

    // Trajectory definition
    // waypoints: num_waypoints x 3 matrix, each row is (x, y, z) in metres
    // Path manager smooths corners using min_turn_radius
    MatrixXd waypoints;
    double mean_velocity;          // m/s — desired speed along trajectory
    double min_turn_radius;        // m — minimum turning radius at waypoints

    // Timing
    double dt;                     // seconds — discretisation timestep
    double t_total;                // seconds — total mission duration
};

// ============================================================
// System parameters — computed from Config by params_build()
// ============================================================

struct SystemParams {
    int N;                         // number of drones
    int n;                         // states per drone (10)
    int m;                         // inputs per drone (4)
    int total_states;              // N * n — dimension of full state vector
    int total_inputs;              // N * m — total control inputs across team
    int num_steps;                 // t_total / dt — number of timesteps in mission

    // Full system dynamics: x_dot = A*x + sum(B_i * u_i)
    MatrixXd A;                    // (N*n x N*n) — block-diagonal, one A_i per drone
    std::vector<MatrixXd> B;       // N matrices, each (N*n x m) — input mapping per drone

    // Cost matrices for each drone (used in cost function J_i, eq 5)
    std::vector<MatrixXd> Q;       // N matrices, each (N*n x N*n) — formation cost
    std::vector<MatrixXd> R;       // N matrices, each (m x m) — control effort cost
    std::vector<MatrixXd> Q_T;     // N matrices, each (N*n x N*n) — terminal cost

    // Communication topology
    MatrixXd D;                    // (N x num_edges) — incidence matrix

    // Precomputed constants — used every timestep in solver, so compute once
    // weighted_Q = sum(alpha_i * Q_i)   — appears in eqs 9b, 9c
    // BR_sum = sum((1/alpha_i) * B_i * R_i^-1 * B_i^T)  — appears in eqs 9b, 9c
    MatrixXd weighted_Q;
    MatrixXd BR_sum;
};

// ============================================================
// Function declarations
// ============================================================

// Main builder — constructs all system matrices from config
SystemParams params_build(const Config& cfg);

// Desired state vector at time t for all drones
// Returns (N*n x 1) vector built from trajectory + formation offsets
// Position states get per-drone offsets, velocity/attitude states are identical
VectorXd get_desired_state(const Config& cfg, double t);
