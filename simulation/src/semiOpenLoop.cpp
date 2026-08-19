#include "simulate.hpp"
#include "computeControl.hpp"
#include <cmath>

void stepCE(Eigen::MatrixXd& C, Eigen::VectorXd& e,
            const RiccatiSolution& ric, const params& P, int k);

SimResult semiOpenLoop(const params& P, const RiccatiSolution& ric, double replan_interval){
	const int n = 10 * P.N;
	const int n_steps = ric.n_steps;
	const int replan_steps = (int)std::lround(replan_interval / P.dt);

	Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n, n_steps);
	Eigen::MatrixXd u = Eigen::MatrixXd::Zero(4*P.N, n_steps);

	Eigen::VectorXd x0_plan = P.x0_flat();
	x.col(0) = x0_plan;

	Eigen::MatrixXd C = Eigen::MatrixXd::Identity(n, n);
	Eigen::VectorXd e = Eigen::VectorXd::Zero(n);

	for(int k = 0; k < n_steps - 1; k++){
		// Ping: resample true state, restart the prediction
		if(k > 0 && k % replan_steps == 0){
			x0_plan = x.col(k);
			C.setIdentity();
			e.setZero();
		}

		Eigen::VectorXd x_hat = C*x0_plan + e;
		Eigen::VectorXd u_k = computeControl(x_hat, ric.s_store[k], ric.p_store[k], P);

		u.col(k)   = u_k;
		x.col(k+1) = x.col(k) + P.dt * (P.A*x.col(k) + P.B*u_k);

		stepCE(C, e, ric, P, k);
	}

	return SimResult{x, u};
}
