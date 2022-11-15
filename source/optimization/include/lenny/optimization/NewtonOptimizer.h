#pragma once

#include <lenny/optimization/Objective.h>

namespace lenny::optimization {

class NewtonOptimizer {
public:
    NewtonOptimizer(const std::string& description = "NewtonOptimizer", double solverResidual = 1e-5);
    ~NewtonOptimizer() = default;

    //Returns whether or not the solver has converged
    bool optimize(Eigen::VectorXd& x, const Objective& objective, uint maxIterations = 1) const;
    void drawGui();

private:
    //Returns gradient norm
    void computeSearchDirection(Eigen::VectorXd& searchDir, const Eigen::VectorXd& x, const Objective& objective) const;
    bool applyLineSearch(Eigen::VectorXd& x, const Eigen::VectorXd& searchDir, const Objective& objective) const;

public:
    std::string description;  //Set by constructor
    double solverResidual;    //Set by constructor

    mutable uint lineSearchMaxIterations = 15;
    double lineSearchStartValue = 1.0;
    double lineSearchMultiplicationFactor = 0.5;

    bool useSparseSolver = true;

    bool printInfos = true;
    bool checkHessianProperties = false;
    bool checkLinearSystemSolve = false;

private:
    //Store infos in class for convenience
    mutable double gradientNorm = HUGE_VALF;
    mutable double objectiveValue = HUGE_VALF;
    mutable double dotProduct = HUGE_VALF;
};

}  // namespace lenny::optimization