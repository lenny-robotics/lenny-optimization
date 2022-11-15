#include <lenny/optimization/Constraint.h>
#include <lenny/tools/Gui.h>
#include <lenny/tools/Logger.h>

namespace lenny::optimization {

Constraint::Constraint(const std::string& description, const double& constraintCheckTolerance)
    : Objective(description), constraintCheckTolerance(constraintCheckTolerance) {}

void Constraint::estimateJacobian(Eigen::SparseMatrixD& pCpX, const Eigen::VectorXd& x) const {
    auto eval = [&](Eigen::VectorXd& C, const Eigen::VectorXd& x) -> void { computeConstraint(C, x); };
    estimateMatrix(pCpX, x, eval, getConstraintNumber(), true);
}

void Constraint::estimateTensor(Eigen::TensorD& p2CpX2, const Eigen::VectorXd& x) const {
    auto eval = [&](Eigen::SparseMatrixD& pCpX, const Eigen::VectorXd& x) -> void { computeJacobian(pCpX, x); };
    tools::FiniteDifference::estimateTensor(p2CpX2, x, eval, getConstraintNumber(), x.size());
}

bool Constraint::testJacobian(const Eigen::VectorXd& x) const {
    auto eval = [&](Eigen::VectorXd& C, const Eigen::VectorXd& x) -> void { computeConstraint(C, x); };
    auto anal = [&](Eigen::SparseMatrixD& pCpX, const Eigen::VectorXd& x) -> void { computeJacobian(pCpX, x); };
    return testMatrix(eval, anal, x, "Jacobian", getConstraintNumber(), true);
}

bool Constraint::testTensor(const Eigen::VectorXd& x) const {
    auto eval = [&](Eigen::SparseMatrixD& pCpX, const Eigen::VectorXd& x) -> void { computeJacobian(pCpX, x); };
    auto anal = [&](Eigen::TensorD& p2CpX2, const Eigen::VectorXd& x) -> void { computeTensor(p2CpX2, x); };
    return tools::FiniteDifference::testTensor(eval, anal, x, "Tensor", getConstraintNumber(), x.size());
}

bool Constraint::checkConstraintSatisfaction(const Eigen::VectorXd& x) const {
    using tools::Logger;

    Eigen::VectorXd C;
    computeConstraint(C, x);
    uint num_c = (uint)C.size();

    if (printConstraintSatisfactionCheck) {
        LENNY_LOG_PRINT(Logger::MAGENTA, "-------------------------------------------------------------\n");
        LENNY_LOG_PRINT(Logger::DEFAULT, "Checking constraint ");
        LENNY_LOG_PRINT(Logger::YELLOW, "%s\n", description.c_str());
    }

    bool checkPassed = true;
    for (uint i = 0; i < num_c; i++) {
        if (getConstraintValueForCheck(C[i]) > constraintCheckTolerance) {
            checkPassed = false;
            if (printConstraintSatisfactionCheck) {
                LENNY_LOG_PRINT(Logger::RED, "\tVIOLATION");
                LENNY_LOG_PRINT(Logger::DEFAULT, " at index %d with value %lf\n", i, C[i]);
            }
        }
    }

    if (printConstraintSatisfactionCheck) {
        if (checkPassed)
            LENNY_LOG_PRINT(Logger::GREEN, "\tPASSED\n");
        LENNY_LOG_PRINT(Logger::DEFAULT, "End of check -> %s\n", description.c_str());
    }

    return checkPassed;
}

void Constraint::drawGuiContent() {
    using tools::Gui;
    Objective::drawGuiContent();
    Gui::I->Input("Constraint Check Tolerance", constraintCheckTolerance);
    Gui::I->Checkbox("Use Tensor For Hessian", useTensorForHessian);
}

}  // namespace lenny::optimization