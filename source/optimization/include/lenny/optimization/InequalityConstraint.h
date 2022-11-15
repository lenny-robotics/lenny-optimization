#pragma once

#include <lenny/optimization/BarrierFunction.h>
#include <lenny/optimization/Constraint.h>

namespace lenny::optimization {

/*
 * C <= 0
 */
class InequalityConstraint : public Constraint {
public:
    //--- Constructor
    InequalityConstraint(const std::string& description);
    virtual ~InequalityConstraint() = default;

    //--- Objective evaluation
    double computeValue(const Eigen::VectorXd& x) const override;
    void computeGradient(Eigen::VectorXd& pVpX, const Eigen::VectorXd& x) const override;
    void computeHessian(Eigen::SparseMatrixD& p2VpX2, const Eigen::VectorXd& x) const override;

protected:
    //--- Constraint check helper
    double getConstraintValueForCheck(const double& C_i) const override;

    //--- Gui
    virtual void drawGuiContent() override;

public:
    BarrierFunction barrier = BarrierFunction(1.0, 0.001);
};

}  // namespace lenny::optimization