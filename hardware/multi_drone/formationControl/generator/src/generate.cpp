#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "params.hpp"
#include "riccatiSolver.hpp"
#include "odes.hpp"

using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static void writeMat(const std::string& fname,
                     const std::vector<Eigen::MatrixXd>& M, int k0){
	std::ofstream f(fname, std::ios::binary);
	for(size_t k = k0; k < M.size(); k++){
		RowMajor RM = M[k];
		f.write(reinterpret_cast<const char*>(RM.data()),
		        RM.size() * sizeof(double));
	}
}

static void writeVec(const std::string& fname,
                     const std::vector<Eigen::VectorXd>& v, int k0){
	std::ofstream f(fname, std::ios::binary);
	for(size_t k = k0; k < v.size(); k++)
		f.write(reinterpret_cast<const char*>(v[k].data()),
		        v[k].size() * sizeof(double));
}

int main(){
	params P;
	const int n = 10 * P.N;

	std::printf("Riccati solver...\n");
	RiccatiSolution ric = riccatiSolver(P);
	const int n_steps = ric.n_steps;
	std::printf("  ||S(t0)||_F = %.10f\n", ric.s_store[0].norm());
	std::printf("  ||p(t0)||   = %.10f\n", ric.p_store[0].norm());

	writeMat("../arrays/s.bin", ric.s_store, 0);
	writeVec("../arrays/p.bin", ric.p_store, 0);

	std::vector<double> replanTimes;
	for(double t = 0.0; t < ric.tT; t += 5.0) replanTimes.push_back(t);

	std::ofstream man("../arrays/manifest.json");
	man.precision(12);
	man << "{\n  \"n\": " << n
	    << ",\n  \"dt\": " << P.dt
	    << ",\n  \"tT\": " << ric.tT
	    << ",\n  \"n_steps\": " << n_steps
	    << ",\n  \"replans\": [\n";

	double total_mb = (double)n_steps * n * n * 8 / 1e6;

	std::printf("\n  %-8s  %8s  %10s\n", "tau(s)", "k0", "c MB");

	for(size_t r = 0; r < replanTimes.size(); r++){
		double tau = replanTimes[r];
		int k0 = (int)std::lround(tau / P.dt);

		Eigen::MatrixXd C = Eigen::MatrixXd::Identity(n, n);
		Eigen::VectorXd e = Eigen::VectorXd::Zero(n);

		std::vector<Eigen::MatrixXd> C_store(n_steps);
		std::vector<Eigen::VectorXd> e_store(n_steps);

		for(int k = k0; k < n_steps; k++){
			C_store[k] = C;
			e_store[k] = e;
			if(k < n_steps - 1) stepCE(C, e, ric, P, k);
		}

		char buf[16];
		std::snprintf(buf, sizeof(buf), "%02d", (int)std::lround(tau));
		writeMat("../arrays/c_" + std::string(buf) + ".bin", C_store, k0);
		writeVec("../arrays/e_" + std::string(buf) + ".bin", e_store, k0);

		double mb = (double)(n_steps - k0) * n * n * 8 / 1e6;
		total_mb += mb;
		std::printf("  %-8.1f  %8d  %10.1f\n", tau, k0, mb);

		man << "    {\"tau\": " << tau << ", \"k0\": " << k0
		    << ", \"steps\": " << (n_steps - k0) << "}"
		    << (r + 1 < replanTimes.size() ? "," : "") << "\n";
	}

	man << "  ]\n}\n";
	std::printf("\n  total ~%.1f MB\n", total_mb);
	return 0;
}
