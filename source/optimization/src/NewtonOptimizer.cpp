#include <lenny/optimization/NewtonOptimizer.h>
#include <lenny/optimization/Utils.h>
#include <lenny/tools/Gui.h>
#include <lenny/tools/Logger.h>

namespace lenny::optimization {

NewtonOptimizer::NewtonOptimizer(const std::string& description, double solverResidual) : description(description), solverResidual(solverResidual) {}

bool NewtonOptimizer::optimize(Eigen::VectorXd& x, const Objective& objective, uint maxIterations) const {
    //Make sure we iterate at least once
    if (maxIterations <= 0)
        maxIterations = 1;

    //Apply optimization loop
    for (uint i = 0; i < maxIterations; i++) {
        //Compute search direction
        Eigen::VectorXd searchDir;
        computeSearchDirection(searchDir, x, objective);

        if (printInfos)
            LENNY_LOG_PRINT(tools::Logger::DEFAULT, "[%s] (Objective value: %10.10lf. Gradient norm: %10.10lf. Dot product: %10.10lf): ", description.c_str(), objectiveValue,
                            gradientNorm, dotProduct)

        //Check convergence
        if (gradientNorm < solverResidual) {
            if (printInfos)
                LENNY_LOG_PRINT(tools::Logger::GREEN, "CONVERGED\n")
            return true;
        }

        //Apply line search
        const bool betterSolutionFound = applyLineSearch(x, searchDir, objective);
        if (printInfos) {
            if (betterSolutionFound)
                LENNY_LOG_PRINT(tools::Logger::BLUE, "IN PROGRESS\n")
            else
                LENNY_LOG_PRINT(tools::Logger::RED, "STUCK\n")
        }
    }

    //Not converged (yet)
    return false;
}

void NewtonOptimizer::drawGui() {
    using tools::Gui;
    if (Gui::I->TreeNode(description.c_str())) {
        Gui::I->Checkbox("Print Infos", printInfos);

        Gui::I->Checkbox("Use Sparse Solver", useSparseSolver);
        Gui::I->Checkbox("Check Hessian Properties", checkHessianProperties);
        Gui::I->Checkbox("Check Linear System Solve", checkLinearSystemSolve);

        Gui::I->Input("Solver Residual", solverResidual);
        Gui::I->Input("Max Line Search Iterations", lineSearchMaxIterations);

        Gui::I->TreePop();
    }
}

template <class HESSIAN_TYPE>
inline void computeSearchDir(Eigen::VectorXd& searchDir, const Eigen::VectorXd& x, const Objective& objective, const Eigen::VectorXd& gradient,
                             const std::string& description, const bool checkHessianProperties, const bool checkLinearSystemSolve) {
    //Compute hessian
    HESSIAN_TYPE hessian;
    objective.computeHessian(hessian, x);

    //Check properties
    if (checkHessianProperties) {
        utils::applyInvertibilityCheck(hessian, description);
        if (!objective.useFullHessian)
            utils::applyLowerTriangularCheck(hessian, description);
    }

    //Solve linear system
    utils::solveLinearSystem(searchDir, hessian, gradient, description, !objective.useFullHessian);
    if (checkLinearSystemSolve)
        utils::applyLinearSystemSolveCheck(searchDir, hessian, gradient, description, !objective.useFullHessian);

    //Apply dynamic regularization
    utils::applyDynamicRegularization(searchDir, hessian, gradient, description, !objective.useFullHessian);
}

void NewtonOptimizer::computeSearchDirection(Eigen::VectorXd& searchDir, const Eigen::VectorXd& x, const Objective& objective) const {
    //Prepare objectives
    objective.preDerivativeEvaluation(x);

    //Compute gradient
    Eigen::VectorXd gradient;
    objective.computeGradient(gradient, x);

    //Compute search direction
    if (useSparseSolver)
        computeSearchDir<Eigen::SparseMatrixD>(searchDir, x, objective, gradient, description, checkHessianProperties, checkLinearSystemSolve);
    else
        computeSearchDir<Eigen::MatrixXd>(searchDir, x, objective, gradient, description, checkHessianProperties, checkLinearSystemSolve);

    //Update log variables
    this->objectiveValue = objective.computeValue(x);
    this->gradientNorm = gradient.norm();
    this->dotProduct = searchDir.dot(gradient);
}

bool NewtonOptimizer::applyLineSearch(Eigen::VectorXd& x, const Eigen::VectorXd& searchDir, const Objective& objective) const {
    //Make sure we iterate at least once
    if (lineSearchMaxIterations <= 0)
        lineSearchMaxIterations = 1;

    //Line search loop
    double alpha = lineSearchStartValue;
    for (uint i = 0; i < lineSearchMaxIterations; i++) {
        //Try new solution
        const Eigen::VectorXd x_new = x - searchDir * alpha;

        //Compute new objective value
        objective.preValueEvaluation(x_new);
        const double newObjectiveValue = objective.computeValue(x_new);

        //If not satisfying, try again with updated alpha
        if (newObjectiveValue > this->objectiveValue) {
            alpha *= lineSearchMultiplicationFactor;
        } else {  //If satisfied, update values and return
            x = x_new;
            return true;  //Better solution found
        }
    }
    return false;  //NO better solution found!
}

}  // namespace lenny::optimization