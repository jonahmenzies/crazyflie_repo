#include "odes.hpp"

Eigen::MatrixXd sODE(const Eigen::MatrixXd& S,
                     const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& AQ_sum,
                     const Eigen::MatrixXd& BR_sum){
	return -S*A - A.transpose()*S - AQ_sum + S*BR_sum*S;
}

Eigen::VectorXd pODE(const Eigen::VectorXd& p,
                     const Eigen::MatrixXd& S,
                     const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& AQ_sum,
                     const Eigen::MatrixXd& BR_sum,
                     const Eigen::VectorXd& xd_t){
	return (S*BR_sum - A.transpose())*p + AQ_sum*xd_t;
}

Eigen::MatrixXd cODE(const Eigen::MatrixXd& C,
                     const Eigen::MatrixXd& S,
                     const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& BR_sum){
	return (A - BR_sum*S) * C;
}

Eigen::VectorXd eODE(const Eigen::VectorXd& e,
                     const Eigen::MatrixXd& S,
                     const Eigen::VectorXd& p,
                     const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& BR_sum){
	return (A - BR_sum*S)*e - BR_sum*p;
}
