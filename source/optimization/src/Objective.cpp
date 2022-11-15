#include <lenny/optimization/Objective.h>
#include <lenny/tools/Gui.h>

namespace lenny::optimization {

Objective::Objective(const std::string& description, const bool useFullHessian) : tools::FiniteDifference(description), useFullHessian(useFullHessian) {}

void Objective::computeHessian(Eigen::MatrixXd& p2VpX2, const Eigen::VectorXd& x) const {
    Eigen::SparseMatrixD p2VpX2_sparse;
    computeHessian(p2VpX2_sparse, x);
    p2VpX2 = p2VpX2_sparse.toDense();
}

void Objective::estimateGradient(Eigen::VectorXd& pVpX, const Eigen::VectorXd& x) const {
    auto eval = [&](const Eigen::VectorXd& x) -> double { return computeValue(x); };
    estimateVector(pVpX, x, eval);
}

void Objective::estimateHessian(Eigen::SparseMatrixD& p2VpX2, const Eigen::VectorXd& x) const {
    auto eval = [&](Eigen::VectorXd& vec, const Eigen::VectorXd& x) -> void { computeGradient(vec, x); };
    estimateMatrix(p2VpX2, x, eval, x.size(), useFullHessian);
}

bool Objective::testGradient(const Eigen::VectorXd& x) const {
    auto eval = [&](const Eigen::VectorXd& x) -> double { return computeValue(x); };
    auto anal = [&](Eigen::VectorXd& vec, const Eigen::VectorXd& x) -> void { computeGradient(vec, x); };
    return testVector(eval, anal, x, "Gradient");
}

bool Objective::testHessian(const Eigen::VectorXd& x) const {
    auto eval = [&](Eigen::VectorXd& vec, const Eigen::VectorXd& x) -> void { computeGradient(vec, x); };
    auto anal = [&](Eigen::SparseMatrixD& sMat, const Eigen::VectorXd& x) -> void { computeHessian(sMat, x); };
    return testMatrix(eval, anal, x, "Hessian", x.size(), useFullHessian);
}

void Objective::drawGui() {
    using tools::Gui;
    if (Gui::I->TreeNode(description.c_str())) {
        drawGuiContent();
        Gui::I->TreePop();
    }
}

}  // namespace lenny::optimization