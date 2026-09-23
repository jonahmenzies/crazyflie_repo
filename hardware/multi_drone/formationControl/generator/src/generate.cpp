#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include "params.hpp"
#include "riccatiSolver.hpp"
#include "odes.hpp"

using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static void writeMat(const std::string& fname,
                     const std::vector<Eigen::MatrixXd>& M){
	std::ofstream f(fname, std::ios::binary);
	for(size_t k = 0; k < M.size(); k++){
		RowMajor RM = M[k];
		f.write(reinterpret_cast<const char*>(RM.data()),
		        RM.size() * sizeof(double));
	}
}

static void writeVec(const std::string& fname,
                     const std::vector<Eigen::VectorXd>& v){
	std::ofstream f(fname, std::ios::binary);
	for(size_t k = 0; k < v.size(); k++)
		f.write(reinterpret_cast<const char*>(v[k].data()),
		        v[k].size() * sizeof(double));
}

static void writeSingle(const std::string& fname, const Eigen::MatrixXd& M){
	std::ofstream f(fname, std::ios::binary);
	RowMajor RM = M;
	f.write(reinterpret_cast<const char*>(RM.data()),
	        RM.size() * sizeof(double));
}

int main(int argc, char** argv){
	std::filesystem::create_directories("../arrays");

	const double replan_interval = (argc > 1) ? std::atof(argv[1]) : 1;

	params P;
	const int n = 10 * P.N;
	const int m = 4 * P.N;
	const int stride = 5;

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

	// G stacks the per-agent gain blocks, so u = -G*(S*x + p).
	// Matches computeControl exactly, minus the clamp, which is nonlinear
	// and has to be applied at flight time instead.
	Eigen::MatrixXd G(m, n);
	G.setZero();
	for(int i = 0; i < P.N; i++){
		Eigen::MatrixXd Bi = P.B.middleCols(i*4, 4);
		G.block(i*4, 0, 4, n) =
			(1.0/P.alpha[i]) * (P.Ri[i].inverse() * Bi.transpose());
	}

	writeMat("../arrays/s.bin", ric.s_store);
	writeVec("../arrays/p.bin", ric.p_store);
	writeSingle("../arrays/xd.bin", ric.xd);

	std::vector<double> replanTimes;
	for(double t = 0.0; t < ric.tT; t += replan_interval) replanTimes.push_back(t);

	std::ofstream man("../arrays/manifest.json");
	man.precision(12);
	man << "{\n  \"n\": " << n
	    << ",\n  \"m\": " << m
	    << ",\n  \"dt\": " << P.dt
	    << ",\n  \"stride\": " << stride
	    << ",\n  \"tT\": " << ric.tT
	    << ",\n  \"n_steps\": " << n_steps
	    << ",\n  \"replan_interval\": " << replan_interval
	    << ",\n  \"mass\": " << P.mass
	    << ",\n  \"g\": " << P.g;

	man << ",\n  \"formation_offsets\": [";
	for(int i = 0; i < P.N; i++)
		man << (i ? ", " : "")
		    << "[" << P.formation_offsets(i,0) << ", "
		            << P.formation_offsets(i,1) << ", "
		            << P.formation_offsets(i,2) << "]";
	man << "]";

	man << ",\n  \"xd0\": [" << ric.xd(0,0) << ", "
	                          << ric.xd(0,3) << ", "
	                          << ric.xd(0,6) << "]";

	man << ",\n  \"replans\": [\n";

	double total_mb = (double)n_steps * n * n * 8 / 1e6;

	std::printf("\n  %-8s  %8s  %8s  %10s  %10s\n",
	            "tau(s)", "k0", "rows", "c MB", "cu MB");

	for(size_t r = 0; r < replanTimes.size(); r++){
		double tau = replanTimes[r];
		int k0 = (int)std::lround(tau / P.dt);

		Eigen::MatrixXd C = Eigen::MatrixXd::Identity(n, n);
		Eigen::VectorXd e = Eigen::VectorXd::Zero(n);

		std::vector<Eigen::MatrixXd> C_store;
		std::vector<Eigen::VectorXd> e_store;
		std::vector<Eigen::MatrixXd> cu_store;
		std::vector<Eigen::VectorXd> eu_store;

		for(int k = k0; k < n_steps; k++){
			if((k - k0) % stride == 0){
				C_store.push_back(C);
				e_store.push_back(e);

				const Eigen::MatrixXd& S = ric.s_store[k];
				const Eigen::VectorXd& p = ric.p_store[k];
				cu_store.push_back(-G * S * C);
				eu_store.push_back(-G * (S * e + p));
			}
			if(k < n_steps - 1) stepCE(C, e, ric, P, k);
		}

		// Name files by replan index, not tau. Rounding tau to an integer
		// collides for any sub-second interval: tau=0.5 and tau=1.0 both
		// round to 1 and the second overwrites the first.
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

		man << "    {\"tau\": " << tau << ", \"k0\": " << k0
		    << ", \"steps\": " << C_store.size() << "}"
		    << (r + 1 < replanTimes.size() ? "," : "") << "\n";
	}

	man << "  ]\n}\n";
	std::printf("\n  total ~%.1f MB\n", total_mb);
	return 0;
}
