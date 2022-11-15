#pragma once

#include <lenny/optimization/Constraint.h>

namespace lenny::optimization {

/*
 * C == 0
 */
class EqualityConstraint : public Constraint {
public:
    //--- Constructor
    EqualityConstraint(const std::string& description);
    virtual ~EqualityConstraint() = default;

    //--- Objective evaluation
    double computeValue(const Eigen::VectorXd& x) const override;
    void computeGradient(Eigen::VectorXd& pVpX, const Eigen::VectorXd& x) const override;
    void computeHessian(Eigen::SparseMatrixD& p2VpX2, const Eigen::VectorXd& x) const override;

protected:
    //--- Constraint check helper
    double getConstraintValueForCheck(const double& C_i) const override;
};

}  // namespace lenny::optimization