#pragma once

#include <lenny/optimization/Objective.h>

namespace lenny::optimization {

class TotalObjective : public Objective {
public:
    TotalObjective(const std::string& description, const double& regularizerWeight = 1e-5);
    TotalObjective(TotalObjective&&) = default;
    virtual ~TotalObjective() = default;

    //--- Evaluation
    virtual double computeValue(const Eigen::VectorXd& x) const override;
    virtual void computeGradient(Eigen::VectorXd& pVpX, const Eigen::VectorXd& x) const override;
    virtual void computeHessian(Eigen::SparseMatrixD& p2VpX2, const Eigen::VectorXd& x) const override;

    //--- Tests
    virtual bool testIndividualFirstDerivatives(const Eigen::VectorXd& x) const;
    virtual bool testIndividualSecondDerivatives(const Eigen::VectorXd& x) const;
    virtual bool testGradient(const Eigen::VectorXd& x) const;
    virtual bool testHessian(const Eigen::VectorXd& x) const;

    //--- Solver
    virtual bool preValueEvaluation(const Eigen::VectorXd& x) const override;
    virtual void preDerivativeEvaluation(const Eigen::VectorXd& x) const override;

    //--- Constraints
    virtual bool checkConstraintSatisfaction(const Eigen::VectorXd& x) const;

    //--- Gui
    virtual void drawGui() override;

public:
    std::vector<std::pair<Objective::UPtr, double>> subObjectives;  //[objective, weight]
    double regularizerWeight;                                       //Set by constructor
};

}  // namespace lenny::optimization
