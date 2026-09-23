#include "computeControl.hpp"
#include <algorithm>
#include <cmath>

static void inclinationProtection(Eigen::VectorXd& u, const params& P){
	for(int i = 0; i < P.N; i++){
		int idx = i*4;
		u(idx+0) = std::clamp(u(idx+0), -M_PI/4.0, M_PI/4.0);          // pitch
		u(idx+1) = std::clamp(u(idx+1), -M_PI/4.0, M_PI/4.0);          // roll
		u(idx+2) = std::clamp(u(idx+2), -P.mass*P.g, 2.0*P.mass*P.g);  // thrust
		u(idx+3) = std::clamp(u(idx+3), -2.0*M_PI, 2.0*M_PI);          // yaw rate
	}
}

Eigen::VectorXd computeControl(const Eigen::VectorXd& x,
                               const Eigen::MatrixXd& S,
                               const Eigen::VectorXd& p,
                               const params& P){
	Eigen::VectorXd sp = S*x + p;
	Eigen::VectorXd u  = Eigen::VectorXd::Zero(4*P.N);

	for(int i = 0; i < P.N; i++){
		Eigen::MatrixXd Bi = P.B.middleCols(i*4, 4);
		u.segment(i*4, 4) = -(1.0/P.alpha[i]) * (P.Ri[i].inverse() * Bi.transpose()) * sp;
	}

	inclinationProtection(u, P);
	return u;
}
