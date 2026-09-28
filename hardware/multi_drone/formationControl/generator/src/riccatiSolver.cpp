#include "riccatiSolver.hpp"
#include "pathManager.hpp"
#include "odes.hpp"
#include <cstdio>

// Solves S(t) and p(t) backwards from the end of the mission with RK4,
// storing both at every timestep.
RiccatiSolution riccatiSolver(const params& P){
	// Desired state of every drone over the whole mission
	Trajectory traj = pathManager(P);
	const int n_steps = traj.xd.rows();
	const Eigen::MatrixXd& xd = traj.xd;

	std::printf("  Mission time tT = %.2f s,  n_steps = %d\n", traj.tT, n_steps);

	const int n = 10 * P.N;

	std::vector<Eigen::MatrixXd> s_store(n_steps);
	std::vector<Eigen::VectorXd> p_store(n_steps);

	// Final conditions at the end of the mission
	Eigen::MatrixXd S = P.AQ_sum;
	Eigen::VectorXd p = -P.AQ_sum * xd.row(n_steps-1).transpose();

	s_store[n_steps-1] = S;
	p_store[n_steps-1] = p;
	const double h = -P.dt;    // negative: stepping backwards in time

	for(int k = n_steps - 1; k > 0; k--){
		// Desired state at the start, middle and end of this step
		Eigen::VectorXd xd_k   = xd.row(k).transpose();
		Eigen::VectorXd xd_km  = xd.row(k-1).transpose();
		Eigen::VectorXd xd_mid = 0.5 * (xd_k + xd_km);

		// The four RK4 slopes
		Eigen::MatrixXd ks1 = sODE(S, P.A, P.AQ_sum, P.BR_sum);
		Eigen::VectorXd kp1 = pODE(p, S, P.A, P.AQ_sum, P.BR_sum, xd_k);

		Eigen::MatrixXd S2 = S + 0.5*h*ks1;
		Eigen::VectorXd p2 = p + 0.5*h*kp1;
		Eigen::MatrixXd ks2 = sODE(S2, P.A, P.AQ_sum, P.BR_sum);
		Eigen::VectorXd kp2 = pODE(p2, S2, P.A, P.AQ_sum, P.BR_sum, xd_mid);

		Eigen::MatrixXd S3 = S + 0.5*h*ks2;
		Eigen::VectorXd p3 = p + 0.5*h*kp2;
		Eigen::MatrixXd ks3 = sODE(S3, P.A, P.AQ_sum, P.BR_sum);
		Eigen::VectorXd kp3 = pODE(p3, S3, P.A, P.AQ_sum, P.BR_sum, xd_mid);

		Eigen::MatrixXd S4 = S + h*ks3;
		Eigen::VectorXd p4 = p + h*kp3;
		Eigen::MatrixXd ks4 = sODE(S4, P.A, P.AQ_sum, P.BR_sum);
		Eigen::VectorXd kp4 = pODE(p4, S4, P.A, P.AQ_sum, P.BR_sum, xd_km);

		S += (h/6.0) * (ks1 + 2*ks2 + 2*ks3 + ks4);
		p += (h/6.0) * (kp1 + 2*kp2 + 2*kp3 + kp4);

		// Keep S symmetric against rounding drift
		S = 0.5 * (S + S.transpose().eval());

		s_store[k-1] = S;
		p_store[k-1] = p;
	}

	return RiccatiSolution{s_store, p_store, xd, traj.tT, n_steps};
}
