#include "odes.hpp"

// The four ODEs of the controller.
// S and p are solved backwards from the end of the mission (riccatiSolver.cpp).
// C and e are solved forwards from each ping (stepCE below), giving the
// predicted state x(t) = C(t) x_meas + e(t).

// dS/dt: the Riccati equation
Eigen::MatrixXd sODE(const Eigen::MatrixXd& S,
                     const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& AQ_sum,
                     const Eigen::MatrixXd& BR_sum){
	return -S*A - A.transpose()*S - AQ_sum + S*BR_sum*S;
}

// dp/dt, including the -A^T p term that Jiang et al. (2020) Eq. 9c is missing
// (corrected from Aghajani & Doustmohammadi, 2015)
Eigen::VectorXd pODE(const Eigen::VectorXd& p,
                     const Eigen::MatrixXd& S,
                     const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& AQ_sum,
                     const Eigen::MatrixXd& BR_sum,
                     const Eigen::VectorXd& xd_t){
	return (S*BR_sum - A.transpose())*p + AQ_sum*xd_t;
}

// dC/dt: closed-loop state transition
Eigen::MatrixXd cODE(const Eigen::MatrixXd& C,
                     const Eigen::MatrixXd& S,
                     const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& BR_sum){
	return (A - BR_sum*S) * C;
}

// de/dt: the part of the predicted state driven by p
Eigen::VectorXd eODE(const Eigen::VectorXd& e,
                     const Eigen::MatrixXd& S,
                     const Eigen::VectorXd& p,
                     const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& BR_sum){
	return (A - BR_sum*S)*e - BR_sum*p;
}

// One RK4 step of C and e, from timestep k to k+1.
// S and p are only stored at whole timesteps, so the half-step values are averages.
void stepCE(Eigen::MatrixXd& C, Eigen::VectorXd& e,
            const RiccatiSolution& ric, const params& P, int k){
	const double h = P.dt;

	// S and p at the start, middle and end of the step
	const Eigen::MatrixXd& S1 = ric.s_store[k];
	const Eigen::MatrixXd& S3 = ric.s_store[k+1];
	Eigen::MatrixXd S2 = 0.5*(S1 + S3);

	const Eigen::VectorXd& p1 = ric.p_store[k];
	const Eigen::VectorXd& p3 = ric.p_store[k+1];
	Eigen::VectorXd p2 = 0.5*(p1 + p3);

	// The four RK4 slopes
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
