// Precomputes everything the flight needs, so the flight loop only does
// matrix-vector multiplies. Writes into ../arrays/:
//   s.bin, p.bin      Riccati solution S(t) and p(t) at every timestep
//   xd.bin            desired state of every drone at every timestep
//   c_NNN, e_NNN      predicted state  x(t) = c(t) x_meas + e(t)   for ping NNN
//   cu_NNN, eu_NNN    control input    u(t) = cu(t) x_meas + eu(t) for ping NNN
//   manifest.json     sizes, timing and settings, read by flight/arrays.py
//
//     ./generate 5    replan every 5 s (default 1 s)
//
// Run from generator/

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include "params.hpp"
#include "riccatiSolver.hpp"
#include "odes.hpp"

// numpy reads the files row by row, so matrices are written row-major
using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Start time, first timestep and number of stored rows for one ping
struct Replan {
	double tau;
	int k0;
	size_t steps;
};


// ---- File writers ----

// Write a list of matrices to one binary file, one after another
static void writeMat(const std::string& fname, const std::vector<Eigen::MatrixXd>& M){
	std::ofstream f(fname, std::ios::binary);
	for(size_t k = 0; k < M.size(); k++){
		RowMajor RM = M[k];
		f.write(reinterpret_cast<const char*>(RM.data()), RM.size() * sizeof(double));
	}
}

// Write a list of vectors to one binary file, one after another
static void writeVec(const std::string& fname, const std::vector<Eigen::VectorXd>& v){
	std::ofstream f(fname, std::ios::binary);
	for(size_t k = 0; k < v.size(); k++)
		f.write(reinterpret_cast<const char*>(v[k].data()), v[k].size() * sizeof(double));
}

// Write one matrix to a binary file
static void writeSingle(const std::string& fname, const Eigen::MatrixXd& M){
	std::ofstream f(fname, std::ios::binary);
	RowMajor RM = M;
	f.write(reinterpret_cast<const char*>(RM.data()), RM.size() * sizeof(double));
}

// Write manifest.json, describing everything in ../arrays/
static void writeManifest(const params& P, const RiccatiSolution& ric,
                          int n, int m, int stride, double replan_interval,
                          const std::vector<Replan>& replans){
	std::ofstream man("../arrays/manifest.json");
	man.precision(12);

	// Sizes, timing and physical constants
	man << "{\n  \"n\": " << n
	    << ",\n  \"m\": " << m
	    << ",\n  \"dt\": " << P.dt
	    << ",\n  \"stride\": " << stride
	    << ",\n  \"tT\": " << ric.tT
	    << ",\n  \"n_steps\": " << ric.n_steps
	    << ",\n  \"replan_interval\": " << replan_interval
	    << ",\n  \"mass\": " << P.mass
	    << ",\n  \"g\": " << P.g;

	// Each drone's offset from the formation centre
	man << ",\n  \"formation_offsets\": [";
	for(int i = 0; i < P.N; i++)
		man << (i ? ", " : "")
		    << "[" << P.formation_offsets(i,0) << ", "
		            << P.formation_offsets(i,1) << ", "
		            << P.formation_offsets(i,2) << "]";
	man << "]";

	// Drone 0's desired position at t = 0
	man << ",\n  \"xd0\": [" << ric.xd(0,0) << ", "
	                          << ric.xd(0,3) << ", "
	                          << ric.xd(0,6) << "]";

	// One entry per ping
	man << ",\n  \"replans\": [\n";
	for(size_t r = 0; r < replans.size(); r++){
		man << "    {\"tau\": " << replans[r].tau << ", \"k0\": " << replans[r].k0
		    << ", \"steps\": " << replans[r].steps << "}"
		    << (r + 1 < replans.size() ? "," : "") << "\n";
	}
	man << "  ]\n}\n";
}


int main(int argc, char** argv){
	std::filesystem::create_directories("../arrays");

	// Replan interval from the command line, default 1 s
	const double replan_interval = (argc > 1) ? std::atof(argv[1]) : 1;

	params P;
	const int n = 10 * P.N;     // state length
	const int m = 4 * P.N;      // control length
	const int stride = 5;       // keep every 5th timestep (20 Hz)

	// ---- Riccati solution over the whole mission ----
	std::printf("Riccati solver...\n");
	RiccatiSolution ric = riccatiSolver(P);
	const int n_steps = ric.n_steps;
	std::printf("  ||S(t0)||_F = %.10f\n", ric.s_store[0].norm());
	std::printf("  ||p(t0)||   = %.10f\n", ric.p_store[0].norm());
	std::printf("  replan interval = %.2f s\n", replan_interval);

	if(!std::isfinite(ric.s_store[0].norm()) ||
	   !std::isfinite(ric.p_store[0].norm())){
		std::printf("\n  ERROR: Riccati solution diverged (nan/inf).\n");
		std::printf("  R_vals are probably too low. Aborting.\n");
		return 1;
	}

	// ---- Control gain ----
	// G stacks each drone's gain block, so u = -G*(S*x + p).
	// The clamp is nonlinear, so it is applied in flight instead (setpoint_map_pwm.py).
	Eigen::MatrixXd G(m, n);
	G.setZero();
	for(int i = 0; i < P.N; i++){
		Eigen::MatrixXd Bi = P.B.middleCols(i*4, 4);
		G.block(i*4, 0, 4, n) = (1.0/P.alpha[i]) * (P.Ri[i].inverse() * Bi.transpose());
	}

	writeMat("../arrays/s.bin", ric.s_store);
	writeVec("../arrays/p.bin", ric.p_store);
	writeSingle("../arrays/xd.bin", ric.xd);

	// ---- Ping start times ----
	std::vector<double> replanTimes;
	for(double t = 0.0; t < ric.tT; t += replan_interval) replanTimes.push_back(t);

	double total_mb = (double)n_steps * n * n * 8 / 1e6;    // starts with the size of s.bin

	std::printf("\n  %-8s  %8s  %8s  %10s  %10s\n",
	            "tau(s)", "k0", "rows", "c MB", "cu MB");

	// ---- c, e, cu, eu for each ping ----
	std::vector<Replan> replans;
	for(size_t r = 0; r < replanTimes.size(); r++){
		double tau = replanTimes[r];
		int k0 = (int)std::lround(tau / P.dt);

		// At the ping x_pred = x_meas, so c starts as identity and e as zero
		Eigen::MatrixXd C = Eigen::MatrixXd::Identity(n, n);
		Eigen::VectorXd e = Eigen::VectorXd::Zero(n);

		std::vector<Eigen::MatrixXd> C_store;
		std::vector<Eigen::VectorXd> e_store;
		std::vector<Eigen::MatrixXd> cu_store;
		std::vector<Eigen::VectorXd> eu_store;

		// Step forward to the end of the mission, keeping every stride-th step
		for(int k = k0; k < n_steps; k++){
			if((k - k0) % stride == 0){
				C_store.push_back(C);
				e_store.push_back(e);

				// u = -G*(S*x_pred + p), with x_pred = C*x_meas + e
				const Eigen::MatrixXd& S = ric.s_store[k];
				const Eigen::VectorXd& p = ric.p_store[k];
				cu_store.push_back(-G * S * C);
				eu_store.push_back(-G * (S * e + p));
			}
			if(k < n_steps - 1) stepCE(C, e, ric, P, k);
		}

		// Files are named by ping number, not tau. Rounding tau would
		// collide for sub-second intervals (0.5 and 1.0 both round to 1).
		char buf[16];
		std::snprintf(buf, sizeof(buf), "%03d", (int)r);
		std::string tag(buf);
		writeMat("../arrays/c_"  + tag + ".bin", C_store);
		writeVec("../arrays/e_"  + tag + ".bin", e_store);
		writeMat("../arrays/cu_" + tag + ".bin", cu_store);
		writeVec("../arrays/eu_" + tag + ".bin", eu_store);

		double mb  = (double)C_store.size()  * n * n * 8 / 1e6;
		double umb = (double)cu_store.size() * m * n * 8 / 1e6;
		total_mb += mb + umb;
		std::printf("  %-8.1f  %8d  %8zu  %10.1f  %10.1f\n",
		            tau, k0, C_store.size(), mb, umb);

		replans.push_back({tau, k0, C_store.size()});
	}

	writeManifest(P, ric, n, m, stride, replan_interval, replans);

	std::printf("\n  total ~%.1f MB\n", total_mb);
	return 0;
}
