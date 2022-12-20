#include <lenny/optimization/Constraint.h>
#include <lenny/tools/Gui.h>
#include <lenny/tools/Logger.h>

namespace lenny::optimization {

Constraint::Constraint(const std::string& description, const double& constraintCheckTolerance)
    : Objective(description), constraintCheckTolerance(constraintCheckTolerance) {}

void Constraint::estimateJacobian(Eigen::SparseMatrixD& pCpX, const Eigen::VectorXd& x) const {
    auto eval = [&](Eigen::VectorXd& C, const Eigen::VectorXd& x) -> void { computeConstraint(C, x); };
    fd.estimateMatrix(pCpX, x, eval, getConstraintNumber(), true);
}

void Constraint::estimateTensor(Eigen::TensorD& p2CpX2, const Eigen::VectorXd& x) const {
    auto eval = [&](Eigen::SparseMatrixD& pCpX, const Eigen::VectorXd& x) -> void { computeJacobian(pCpX, x); };
    fd.estimateTensor(p2CpX2, x, eval, getConstraintNumber(), x.size());
}

bool Constraint::testJacobian(const Eigen::VectorXd& x) const {
    fdCheckIsBeingApplied = true;
    auto eval = [&](Eigen::VectorXd& C, const Eigen::VectorXd& x) -> void { computeConstraint(C, x); };
    auto anal = [&](Eigen::SparseMatrixD& pCpX, const Eigen::VectorXd& x) -> void { computeJacobian(pCpX, x); };
    const bool successful = fd.testMatrix(eval, anal, x, "Jacobian", getConstraintNumber(), true);
    fdCheckIsBeingApplied = false;
    return successful;
}

bool Constraint::testTensor(const Eigen::VectorXd& x) const {
    fdCheckIsBeingApplied = true;
    auto eval = [&](Eigen::SparseMatrixD& pCpX, const Eigen::VectorXd& x) -> void { computeJacobian(pCpX, x); };
    auto anal = [&](Eigen::TensorD& p2CpX2, const Eigen::VectorXd& x) -> void { computeTensor(p2CpX2, x); };
    const bool successful = fd.testTensor(eval, anal, x, "Tensor", getConstraintNumber(), x.size());
    fdCheckIsBeingApplied = false;
    return successful;
}

void Constraint::checkConstraintSatisfaction(std::vector<uint>& indices, const Eigen::VectorXd& x, const bool& print) const {
    using tools::Logger;

    if (print) {
        LENNY_LOG_PRINT(Logger::MAGENTA, "-------------------------------------------------------------\n");
        LENNY_LOG_PRINT(Logger::DEFAULT, "Checking constraint ");
        LENNY_LOG_PRINT(Logger::YELLOW, "%s\n", description.c_str());
    }

    indices.clear();

    Eigen::VectorXd C;
    computeConstraint(C, x);
    const uint num_c = (uint)C.size();
    for (uint i = 0; i < num_c; i++) {
        if (getConstraintValueForCheck(C[i]) > constraintCheckTolerance) {
            indices.emplace_back(i);
            if (print) {
                LENNY_LOG_PRINT(Logger::RED, "\tVIOLATION");
                LENNY_LOG_PRINT(Logger::DEFAULT, " at index %d with value %lf\n", i, C[i]);
            }
        }
    }

    if (print) {
        if (indices.size() == 0)
            LENNY_LOG_PRINT(Logger::GREEN, "\tPASSED\n");
        LENNY_LOG_PRINT(Logger::DEFAULT, "End of check -> %s\n", description.c_str());
        LENNY_LOG_PRINT(Logger::MAGENTA, "-------------------------------------------------------------\n");
    }
}

void Constraint::checkSoftificationWeights() const {
    if (softificationWeights.size() != getConstraintNumber()) {
        LENNY_LOG_DEBUG("Resetting all softification weights of constraint `%s` to one", description.c_str())
        softificationWeights.setOnes(getConstraintNumber());
    }
}

void Constraint::drawGuiContent() {
    using tools::Gui;
    Objective::drawGuiContent();
    Gui::I->Input("Constraint Check Tolerance", constraintCheckTolerance);
    Gui::I->Checkbox("Use Tensor For Hessian", useTensorForHessian);
    Gui::I->Input("Softification Weights", softificationWeights);
}

}  // namespace lenny::optimization