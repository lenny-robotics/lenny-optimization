#pragma once

#include <lenny/tools/FiniteDifference.h>

#include <memory>

namespace lenny::optimization {

class Objective : public tools::FiniteDifference {
public:
    //--- Pointers
    typedef std::unique_ptr<Objective> UPtr;
    typedef std::shared_ptr<Objective> SPtr;

    //--- Constructor
    Objective(const std::string& description, const bool useFullHessian = false);
    virtual ~Objective() = default;

    //--- Evaluation
    virtual double computeValue(const Eigen::VectorXd& x) const = 0;
    virtual void computeGradient(Eigen::VectorXd& pVpX, const Eigen::VectorXd& x) const = 0;
    virtual void computeHessian(Eigen::SparseMatrixD& p2VpX2, const Eigen::VectorXd& x) const = 0;

    virtual void computeHessian(Eigen::MatrixXd& p2VpX2, const Eigen::VectorXd& x) const;

    //--- Estimation
    void estimateGradient(Eigen::VectorXd& pVpX, const Eigen::VectorXd& x) const;
    void estimateHessian(Eigen::SparseMatrixD& p2VpX2, const Eigen::VectorXd& x) const;

    //--- Tests
    virtual bool testGradient(const Eigen::VectorXd& x) const;
    virtual bool testHessian(const Eigen::VectorXd& x) const;

    //--- Solver
    virtual bool preValueEvaluation(const Eigen::VectorXd& x) const;  //Return false if we should move on with line search
    virtual void preDerivativeEvaluation(const Eigen::VectorXd& x) const {}

    //--- Gui
    virtual void drawGui();

protected:
    virtual void drawGuiContent() {}

public:
    const bool useFullHessian;  //Set by constructor
};

}  // namespace lenny::optimization