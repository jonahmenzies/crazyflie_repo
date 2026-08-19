#include "simulate.hpp"
#include "computeControl.hpp"
#include "odes.hpp"

// Advance C and e one timestep, from grid point k to k+1
void stepCE(Eigen::MatrixXd& C, Eigen::VectorXd& e,
            const RiccatiSolution& ric, const params& P, int k){
	const double h = P.dt;

	const Eigen::MatrixXd& S1 = ric.s_store[k];
	const Eigen::MatrixXd& S3 = ric.s_store[k+1];
	Eigen::MatrixXd S2 = 0.5*(S1 + S3);

	const Eigen::VectorXd& p1 = ric.p_store[k];
	const Eigen::VectorXd& p3 = ric.p_store[k+1];
	Eigen::VectorXd p2 = 0.5*(p1 + p3);

	Eigen::MatrixXd kc1 = cODE(C, S1, P.A, P.BR_sum);
	Eigen::VectorXd ke1 = eODE(e, S1, p1, P.A, P.BR_sum);

	Eigen::MatrixXd kc2 = cODE(C + 0.5*h*kc1, S2, P.A, P.BR_sum);
	Eigen::VectorXd ke2 = eODE(e + 0.5*h*ke1, S2, p2, P.A, P.BR_sum);

	Eigen::MatrixXd kc3 = cODE(C + 0.5*h*kc2, S2, P.A, P.BR_sum);
	Eigen::VectorXd ke3 = eODE(e + 0.5*h*ke2, S2, p2, P.A, P.BR_sum);

	Eigen::MatrixXd kc4 = cODE(C + h*kc3, S3, P.A, P.BR_sum);
	Eigen::VectorXd ke4 = eODE(e + h*ke3, S3, p3, P.A, P.BR_sum);

	C += (h/6.0) * (kc1 + 2*kc2 + 2*kc3 + kc4);
	e += (h/6.0) * (ke1 + 2*ke2 + 2*ke3 + ke4);
}

SimResult openLoop(const params& P, const RiccatiSolution& ric){
	const int n = 10 * P.N;
	const int n_steps = ric.n_steps;

	Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n, n_steps);
	Eigen::MatrixXd u = Eigen::MatrixXd::Zero(4*P.N, n_steps);

	Eigen::VectorXd x0 = P.x0_flat();
	x.col(0) = x0;

	Eigen::MatrixXd C = Eigen::MatrixXd::Identity(n, n);
	Eigen::VectorXd e = Eigen::VectorXd::Zero(n);

	for(int k = 0; k < n_steps - 1; k++){
		Eigen::VectorXd x_hat = C*x0 + e;
		Eigen::VectorXd u_k = computeControl(x_hat, ric.s_store[k], ric.p_store[k], P);

		u.col(k)   = u_k;
		x.col(k+1) = x.col(k) + P.dt * (P.A*x.col(k) + P.B*u_k);

		stepCE(C, e, ric, P, k);
	}

	return SimResult{x, u};
}
