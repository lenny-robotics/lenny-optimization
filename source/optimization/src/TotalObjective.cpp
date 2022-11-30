#include <lenny/optimization/EqualityConstraint.h>
#include <lenny/optimization/InequalityConstraint.h>
#include <lenny/optimization/TotalObjective.h>
#include <lenny/optimization/Utils.h>
#include <lenny/tools/Gui.h>

namespace lenny::optimization {

TotalObjective::TotalObjective(const std::string& description, const double& regularizerWeight)
    : Objective(description), regularizerWeight(regularizerWeight) {}

double TotalObjective::computeValue(const Eigen::VectorXd& x) const {
    double value = 0.0;
    for (const auto& [objective, weight] : subObjectives)
        if (weight > 0.0)
            value += weight * objective->computeValue(x);
    return value;
}

void TotalObjective::computeGradient(Eigen::VectorXd& pVpX, const Eigen::VectorXd& x) const {
    pVpX.resize(x.size());
    pVpX.setZero();
    for (const auto& [objective, weight] : subObjectives) {
        if (weight > 0.0) {
            Eigen::VectorXd pVpX_obj = Eigen::VectorXd::Zero(x.size());
            objective->computeGradient(pVpX_obj, x);
            pVpX += weight * pVpX_obj;
        }
    }
}

void TotalObjective::computeHessian(Eigen::SparseMatrixD& p2VpX2, const Eigen::VectorXd& x) const {
    p2VpX2.resize(x.size(), x.size());
    p2VpX2.setZero();
    for (const auto& [objective, weight] : subObjectives) {
        if (weight > 0.0) {
            Eigen::SparseMatrixD p2VpX2_obj = Eigen::SparseMatrixD(x.size(), x.size());
            p2VpX2_obj.setZero();
            objective->computeHessian(p2VpX2_obj, x);
            p2VpX2 += weight * p2VpX2_obj;
        }
    }

    if (!fdCheckIsBeingApplied)
        utils::applyStaticRegularization(p2VpX2, regularizerWeight);
}

bool TotalObjective::testIndividualFirstDerivatives(const Eigen::VectorXd& x) const {
    bool testSuccessful = true;
    for (const auto& [objective, weight] : subObjectives) {
        if (const InequalityConstraint* iec = dynamic_cast<InequalityConstraint*>(objective.get())) {
            if (!iec->testJacobian(x))
                testSuccessful = false;
        } else if (const EqualityConstraint* ec = dynamic_cast<EqualityConstraint*>(objective.get())) {
            if (!ec->testJacobian(x))
                testSuccessful = false;
        } else if (const Objective* of = dynamic_cast<Objective*>(objective.get())) {
            if (!of->testGradient(x))
                testSuccessful = false;
        }
    }
    return testSuccessful;
}

bool TotalObjective::testIndividualSecondDerivatives(const Eigen::VectorXd& x) const {
    bool testSuccessful = true;
    for (const auto& [objective, weight] : subObjectives) {
        if (const InequalityConstraint* iec = dynamic_cast<InequalityConstraint*>(objective.get())) {
            if (!iec->testTensor(x))
                testSuccessful = false;
        } else if (const EqualityConstraint* ec = dynamic_cast<EqualityConstraint*>(objective.get())) {
            if (!ec->testTensor(x))
                testSuccessful = false;
        } else if (const Objective* of = dynamic_cast<Objective*>(objective.get())) {
            if (!of->testHessian(x))
                testSuccessful = false;
        }
    }
    return testSuccessful;
}

void TotalObjective::setFDCheckIsBeingApplied(bool isBeingApplied) const {
    for (const auto& [objective, weight] : subObjectives)
        objective->setFDCheckIsBeingApplied(isBeingApplied);
    this->fdCheckIsBeingApplied = isBeingApplied;
}

void TotalObjective::preFDEvaluation(const Eigen::VectorXd& x) const {
    for (const auto& [objective, weight] : subObjectives)
        objective->preFDEvaluation(x);
}

bool TotalObjective::preValueEvaluation(const Eigen::VectorXd& x) const {
    bool success = true;
    for (const auto& [objective, weight] : subObjectives)
        if (!objective->preValueEvaluation(x))
            success = false;
    return success;
}

void TotalObjective::preDerivativeEvaluation(const Eigen::VectorXd& x) const {
    for (const auto& [objective, weight] : subObjectives)
        objective->preDerivativeEvaluation(x);
}

bool TotalObjective::checkConstraintSatisfaction(const Eigen::VectorXd& x) const {
    bool checkPassed = true;
    for (const auto& [objective, weight] : subObjectives) {
        if (const InequalityConstraint* iec = dynamic_cast<InequalityConstraint*>(objective.get())) {
            if (!iec->checkConstraintSatisfaction(x))
                checkPassed = false;
        } else if (const EqualityConstraint* ec = dynamic_cast<EqualityConstraint*>(objective.get())) {
            if (!ec->checkConstraintSatisfaction(x))
                checkPassed = false;
        }
    }
    return checkPassed;
}

void TotalObjective::drawGui() {
    using tools::Gui;
    if (Gui::I->TreeNode(description.c_str())) {
        Gui::I->PushItemWidth(100.f);
        if (Gui::I->TreeNode("Weights")) {
            for (auto& [objective, weight] : subObjectives)
                Gui::I->Input(objective->description.c_str(), weight);
            Gui::I->Input("Regularizer", regularizerWeight);

            Gui::I->TreePop();
        }
        Gui::I->PopItemWidth();

        for (auto& [objective, weight] : subObjectives)
            objective->drawGui();

        Gui::I->TreePop();
    }
}

}  // namespace lenny::optimization