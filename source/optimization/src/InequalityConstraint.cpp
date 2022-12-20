#include <lenny/optimization/InequalityConstraint.h>
#include <lenny/tools/Gui.h>
#include <lenny/tools/Utils.h>

namespace lenny::optimization {

InequalityConstraint::InequalityConstraint(const std::string& description) : Constraint(description, 0.0) {}

double InequalityConstraint::computeValue(const Eigen::VectorXd& x) const {
    Eigen::VectorXd C;
    computeConstraint(C, x);
    checkSoftificationWeights();
    double value = 0.0;
    for (uint i = 0; i < C.size(); i++)
        value += softificationWeights[i] * barrier.computeValue(C[i]);
    return value;
}

void InequalityConstraint::computeGradient(Eigen::VectorXd& pVpX, const Eigen::VectorXd& x) const {
    Eigen::VectorXd C;
    computeConstraint(C, x);

    Eigen::SparseMatrixD pCpX;
    computeJacobian(pCpX, x);

    checkSoftificationWeights();
    pVpX.resize(x.size());
    pVpX.setZero();
    for (uint i = 0; i < C.size(); i++)
        pVpX += softificationWeights[i] * barrier.computeFirstDerivative(C[i]) * pCpX.row(i);
}

void InequalityConstraint::computeHessian(Eigen::SparseMatrixD& p2VpX2, const Eigen::VectorXd& x) const {
    Eigen::VectorXd C;
    computeConstraint(C, x);

    Eigen::SparseMatrixD pCpX;
    computeJacobian(pCpX, x);

    checkSoftificationWeights();
    Eigen::TripletDList p2BpX2_entries;
    for (uint i = 0; i < C.size(); i++)
        tools::utils::addTripletDToList(p2BpX2_entries, i, i, softificationWeights[i] * barrier.computeSecondDerivative(C[i]));
    Eigen::SparseMatrixD p2BpX2(C.size(), C.size());
    p2BpX2.setFromTriplets(p2BpX2_entries.begin(), p2BpX2_entries.end());

    if (useFullHessian)
        p2VpX2 = pCpX.transpose() * p2BpX2 * pCpX;
    else
        p2VpX2 = (pCpX.transpose() * p2BpX2 * pCpX).triangularView<Eigen::Lower>();

    if (useTensorForHessian || fdCheckIsBeingApplied) {
        Eigen::TensorD p2CpX2(Eigen::Vector3i(C.size(), x.size(), x.size()));
        computeTensor(p2CpX2, x);
        if (p2CpX2.getNumberOfEntries() > 0) {
            Eigen::VectorXd pBpX(C.size());
            for (uint i = 0; i < C.size(); i++)
                pBpX[i] = softificationWeights[i] * barrier.computeFirstDerivative(C[i]);

            Eigen::SparseMatrixD mat;
            p2CpX2.multiply(mat, pBpX);

            p2VpX2 += useFullHessian ? mat : mat.triangularView<Eigen::Lower>();
        }
    }
}

double InequalityConstraint::getConstraintValueForCheck(const double& C_i) const {
    return C_i;
}

void InequalityConstraint::drawGuiContent() {
    Constraint::drawGuiContent();
    barrier.drawGui();
}

}  // namespace lenny::optimization