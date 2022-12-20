#pragma once

#include <lenny/optimization/Objective.h>

namespace lenny::optimization {

class Constraint : public Objective {
public:
    //--- Constructor
    Constraint(const std::string& description, const double& constraintCheckTolerance);
    virtual ~Constraint() = default;

    //--- Evaluation
    virtual uint getConstraintNumber() const = 0;
    virtual void computeConstraint(Eigen::VectorXd& C, const Eigen::VectorXd& x) const = 0;
    virtual void computeJacobian(Eigen::SparseMatrixD& pCpX, const Eigen::VectorXd& x) const = 0;
    virtual void computeTensor(Eigen::TensorD& p2CpX2, const Eigen::VectorXd& x) const = 0;

    //--- Estimation
    void estimateJacobian(Eigen::SparseMatrixD& pCpX, const Eigen::VectorXd& x) const;
    void estimateTensor(Eigen::TensorD& p2CpX2, const Eigen::VectorXd& x) const;

    //--- Tests
    virtual bool testJacobian(const Eigen::VectorXd& x) const;
    virtual bool testTensor(const Eigen::VectorXd& x) const;

    //--- Check (returns indices with violated constraints)
    virtual void checkConstraintSatisfaction(std::vector<uint>& indices, const Eigen::VectorXd& x, const bool& print) const;

    //--- Helpers
    void checkSoftificationWeights() const;

protected:
    //--- Constraint check helper
    virtual double getConstraintValueForCheck(const double& C_i) const = 0;

    //--- Gui
    virtual void drawGuiContent() override;

public:
    double constraintCheckTolerance;  //Set by constructor
    bool useTensorForHessian = true;
    mutable Eigen::VectorXd softificationWeights;
};

}  // namespace lenny::optimization