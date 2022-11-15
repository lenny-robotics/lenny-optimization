#include <lenny/optimization/EqualityConstraint.h>
#include <lenny/tools/Utils.h>

namespace lenny::optimization {

EqualityConstraint::EqualityConstraint(const std::string& description) : Constraint(description, 1e-5) {}

double EqualityConstraint::computeValue(const Eigen::VectorXd& x) const {
    Eigen::VectorXd C;
    computeConstraint(C, x);
    return 0.5 * C.dot(C);
}

void EqualityConstraint::computeGradient(Eigen::VectorXd& pVpX, const Eigen::VectorXd& x) const {
    Eigen::VectorXd C;
    computeConstraint(C, x);

    Eigen::SparseMatrixD pCpX;
    computeJacobian(pCpX, x);

    pVpX = pCpX.transpose() * C;
}

void EqualityConstraint::computeHessian(Eigen::SparseMatrixD& p2VpX2, const Eigen::VectorXd& x) const {
    Eigen::SparseMatrixD pCpX;
    computeJacobian(pCpX, x);

    if (useFullHessian)
        p2VpX2 = pCpX.transpose() * pCpX;
    else
        p2VpX2 = (pCpX.transpose() * pCpX).triangularView<Eigen::Lower>();

    if (useTensorForHessian || fdCheckIsBeingApplied) {
        Eigen::TensorD p2CpX2(Eigen::Vector3i(pCpX.rows(), pCpX.cols(), x.size()));
        computeTensor(p2CpX2, x);

        if (p2CpX2.getNumberOfEntries() > 0) {
            Eigen::VectorXd C;
            computeConstraint(C, x);

            Eigen::SparseMatrixD mat;
            p2CpX2.multiply(mat, C);

            p2VpX2 += useFullHessian ? mat : mat.triangularView<Eigen::Lower>();
        }
    }
}

double EqualityConstraint::getConstraintValueForCheck(const double& C_i) const {
    return fabs(C_i);
}

}  // namespace lenny::optimization