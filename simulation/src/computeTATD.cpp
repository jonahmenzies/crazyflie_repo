#include "computeTATD.hpp"
#include <cmath>

// Sum of squared position error across all drones at step k
static double posErrSq(const Eigen::MatrixXd& x,
                       const Eigen::MatrixXd& xd,
                       int k, int N){
	double sum = 0.0;
	for(int i = 0; i < N; i++){
		int base = i*10;
		for(int c : {0, 3, 6}){          // x, y, z
			double e = x(base+c, k) - xd(k, base+c);
			sum += e*e;
		}
	}
	return sum;
}

double computeTATD20(const Eigen::MatrixXd& x,
                     const Eigen::MatrixXd& xd,
                     const params& P,
                     double frac){
	const int n_steps = x.cols();
	const int start_k = (int)std::lround((1.0 - frac) * n_steps);
	const int NT      = n_steps - start_k;

	double sum_sq = 0.0;
	for(int k = start_k; k < n_steps; k++){
		sum_sq += posErrSq(x, xd, k, P.N);
	}

	return std::sqrt(sum_sq / (P.N * NT));
}

Eigen::VectorXd computeTATDInstantaneous(const Eigen::MatrixXd& x,
                                         const Eigen::MatrixXd& xd,
                                         const params& P){
	const int n_steps = x.cols();
	Eigen::VectorXd TATD_t(n_steps);
	for(int k = 0; k < n_steps; k++){
		TATD_t(k) = std::sqrt(posErrSq(x, xd, k, P.N) / (P.N * 3));
	}
	return TATD_t;
}
