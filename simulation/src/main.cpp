#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "params.hpp"
#include "riccatiSolver.hpp"
#include "simulate.hpp"
#include "computeTATD.hpp"

static void writeCSV(const std::string& fname,
                     const Eigen::MatrixXd& x,
                     const Eigen::MatrixXd& xd,
                     const params& P){
	std::ofstream f(fname);
	f << "t";
	for(int i = 0; i < P.N; i++) f << ",x" << i << ",y" << i;
	f << ",xd,yd\n";

	for(int k = 0; k < x.cols(); k++){
		f << k * P.dt;
		for(int i = 0; i < P.N; i++){
			f << "," << x(i*10 + 0, k) << "," << x(i*10 + 3, k);
		}
		f << "," << xd(k, 0) << "," << xd(k, 3) << "\n";
	}
}

int main(){
	params P;

	std::printf("Riccati solver...\n");
	RiccatiSolution ric = riccatiSolver(P);
	std::printf("  ||S(t0)||_F = %.10f\n", ric.s_store[0].norm());
	std::printf("  ||p(t0)||   = %.10f\n", ric.p_store[0].norm());

	std::printf("Closed-loop...\n");
	SimResult cl = closedLoop(P, ric);
	double TATD_cl = computeTATD20(cl.x, ric.xd, P);
	std::printf("  TATD20 = %.6f m\n", TATD_cl);
	writeCSV("closedLoop.csv", cl.x, ric.xd, P);

	std::printf("Open-loop...\n");
	SimResult ol = openLoop(P, ric);
	std::printf("  TATD20 = %.6f m\n", computeTATD20(ol.x, ric.xd, P));
	writeCSV("openLoop.csv", ol.x, ric.xd, P);

	std::vector<double> replanTimes = {1, 5, 7, 10, 15, 20};

	std::printf("\n  %-10s  %10s  %12s  %12s\n", "Replan(s)", "#replans", "TATD20 (m)", "%% vs CL");
	for(double Ts : replanTimes){
		SimResult sol = semiOpenLoop(P, ric, Ts);
		double t = computeTATD20(sol.x, ric.xd, P);
		int n_replans = (int)std::ceil(ric.tT / Ts);
		std::printf("  %-10.2f  %10d  %12.6f  %+11.2f%%\n",
		            Ts, n_replans, t, (t - TATD_cl)/TATD_cl * 100.0);
		writeCSV("semiOpenLoop_" + std::to_string((int)Ts) + "s.csv", sol.x, ric.xd, P);
	}

	return 0;
}
