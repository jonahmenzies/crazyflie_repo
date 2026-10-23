// Runs the existing sims (simulation/src) from a real flight's starting state.
// Params, path and Riccati solution come from generator/, so the reference
// is the one the drones flew.
//
//     simflight/simflight <x0.csv> <out_prefix> <interval> [interval ...]
//
// x0.csv holds the 10N starting states (state order per drone, one line).
// Writes <out_prefix>_closed.csv, <out_prefix>_open.csv and
// <out_prefix>_semi_<interval>.csv. Columns: t, then x, y, z for each drone,
// then rx, ry, rz for each drone (the reference). Every 5th timestep is kept
// so the rows line up with the flight log.
// Run from formationControl/

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "params.hpp"
#include "riccatiSolver.hpp"
#include "simulate.hpp"

static const int STRIDE = 5;

// Read the starting state, comma or whitespace separated
static Eigen::VectorXd readX0(const char* path, int n){
	std::ifstream f(path);
	if(!f){
		std::fprintf(stderr, "ERROR: cannot open %s\n", path);
		std::exit(1);
	}
	std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	for(char& c : text) if(c == ',') c = ' ';

	std::istringstream in(text);
	std::vector<double> v;
	double d;
	while(in >> d) v.push_back(d);

	if((int)v.size() != n){
		std::fprintf(stderr, "ERROR: %s has %zu values, need %d\n", path, v.size(), n);
		std::exit(1);
	}
	return Eigen::Map<Eigen::VectorXd>(v.data(), n);
}

// Positions of every drone and of the reference, every STRIDE-th timestep
static void writeCSV(const std::string& fname, const Eigen::MatrixXd& x,
                     const Eigen::MatrixXd& xd, const params& P){
	std::ofstream f(fname);
	f << "t";
	for(int i = 0; i < P.N; i++) f << ",x" << i << ",y" << i << ",z" << i;
	for(int i = 0; i < P.N; i++) f << ",rx" << i << ",ry" << i << ",rz" << i;
	f << "\n";

	f.precision(8);
	for(int k = 0; k < x.cols(); k += STRIDE){
		f << k * P.dt;
		for(int i = 0; i < P.N; i++)
			for(int c : {0, 3, 6}) f << "," << x(i*10 + c, k);
		for(int i = 0; i < P.N; i++)
			for(int c : {0, 3, 6}) f << "," << xd(k, i*10 + c);
		f << "\n";
	}
}

int main(int argc, char** argv){
	if(argc < 4){
		std::fprintf(stderr, "usage: %s <x0.csv> <out_prefix> <interval> [interval ...]\n", argv[0]);
		return 1;
	}

	params P;
	P.x0 = readX0(argv[1], 10 * P.N);
	const std::string prefix = argv[2];

	RiccatiSolution ric = riccatiSolver(P);

	writeCSV(prefix + "_closed.csv", closedLoop(P, ric).x, ric.xd, P);
	writeCSV(prefix + "_open.csv",   openLoop(P, ric).x,   ric.xd, P);

	for(int a = 3; a < argc; a++){
		double Ts = std::atof(argv[a]);
		writeCSV(prefix + "_semi_" + argv[a] + ".csv",
		         semiOpenLoop(P, ric, Ts).x, ric.xd, P);
	}
	return 0;
}
