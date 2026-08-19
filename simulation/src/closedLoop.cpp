#include "simulate.hpp"
#include "computeControl.hpp"

SimResult closedLoop(const params& P, const RiccatiSolution& ric){
	const int n = 10 * P.N;
	const int n_steps = ric.n_steps;

	Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n, n_steps);
	Eigen::MatrixXd u = Eigen::MatrixXd::Zero(4*P.N, n_steps);

	x.col(0) = P.x0_flat();

	for(int k = 0; k < n_steps - 1; k++){
		Eigen::VectorXd u_k = computeControl(x.col(k), ric.s_store[k], ric.p_store[k], P);

		u.col(k)   = u_k;
		x.col(k+1) = x.col(k) + P.dt * (P.A*x.col(k) + P.B*u_k);
	}

	return SimResult{x, u};
}
