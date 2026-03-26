// solver.h
#pragma once
#include "params.h"

// ============================================================
// Solver result — precomputed lookup tables
// ============================================================

struct SolverResult {
    std::vector<MatrixXd> s_table;   // s at each timestep (each N*n x N*n)
    std::vector<VectorXd> p_table;   // p at each timestep (each N*n x 1)
    int num_steps;
    double dt;
};

// ============================================================
// Solver functions
// ============================================================

/// @brief Precompute s(t) and p(t) for the entire mission
/// Solves eq 9b backwards for s, then eq 9c backwards for p
/// @param sys System matrices from params_build
/// @param cfg Mission configuration
/// @return Lookup tables for s and p at every timestep
SolverResult solver_precompute(const SystemParams& sys, const Config& cfg);

/// @brief Save precomputed tables to file for reuse
/// @param result The solved tables
/// @param filename Path to save to
void solver_save(const SolverResult& result, const std::string& filename);

/// @brief Load precomputed tables from file
/// @param filename Path to load from
/// @return Previously solved tables
SolverResult solver_load(const std::string& filename);
