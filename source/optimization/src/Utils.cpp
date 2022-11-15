#include <lenny/optimization/Utils.h>
#include <lenny/tools/Logger.h>

namespace lenny::optimization::utils {

template <class TYPE_Y, class TYPE_A, class TYPE_X, class TYPE_SOLVER>
inline bool solveLinearSystem(TYPE_Y& y, const TYPE_A& A, const TYPE_X& x, const std::string& info) {
    TYPE_SOLVER solver;
    solver.compute(A);
    y = solver.solve(x);

    if (solver.info() != Eigen::Success) {
        LENNY_LOG_WARNING("%s: solve unsuccessful!", info.c_str());
        return false;
    }
    return true;
}

bool solveLinearSystem(Eigen::VectorXd& y, const Eigen::SparseMatrixD& A, const Eigen::VectorXd& x, const std::string& description, bool isLowerTriangular) {
    if (isLowerTriangular)
        return solveLinearSystem<Eigen::VectorXd, Eigen::SparseMatrixD, Eigen::VectorXd, Eigen::SimplicialLDLT<Eigen::SparseMatrixD, Eigen::Lower>>(
            y, A, x, description + "(y = A^-1 * x - sparse + lower triagonal)");
    return solveLinearSystem<Eigen::VectorXd, Eigen::SparseMatrixD, Eigen::VectorXd, Eigen::SparseLU<Eigen::SparseMatrixD>>(
        y, A, x, description + "(y = A^-1 * x - sparse + full matrix)");
}

bool solveLinearSystem(Eigen::VectorXd& y, const Eigen::MatrixXd& A, const Eigen::VectorXd& x, const std::string& description, bool isLowerTriangular) {
    if (isLowerTriangular)
        return solveLinearSystem<Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::LDLT<Eigen::MatrixXd, Eigen::Lower>>(
            y, A, x, description + "(y = A^-1 * x - dense + lower triagonal)");
    return solveLinearSystem<Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::LDLT<Eigen::MatrixXd>>(
        y, A, x, description + "(y = A^-1 * x - dense + full matrix)");
}

bool solveLinearSystem(Eigen::MatrixXd& Y, const Eigen::SparseMatrixD& A, const Eigen::SparseMatrixD& X, const std::string& description,
                       bool isLowerTriangular) {
    if (isLowerTriangular)
        return solveLinearSystem<Eigen::MatrixXd, Eigen::SparseMatrixD, Eigen::SparseMatrixD, Eigen::SimplicialLDLT<Eigen::SparseMatrixD, Eigen::Lower>>(
            Y, A, X, description + "(Y = A^-1 * X - sparse + lower triagonal)");
    return solveLinearSystem<Eigen::MatrixXd, Eigen::SparseMatrixD, Eigen::SparseMatrixD, Eigen::SparseLU<Eigen::SparseMatrixD>>(
        Y, A, X, description + "(Y = A^-1 * X - sparse + full matrix)");
}

bool solveLinearSystem(Eigen::MatrixXd& Y, const Eigen::MatrixXd& A, const Eigen::MatrixXd& X, const std::string& description, bool isLowerTriangular) {
    if (isLowerTriangular)
        return solveLinearSystem<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::LDLT<Eigen::MatrixXd, Eigen::Lower>>(
            Y, A, X, description + "(Y = A^-1 * X - dense + lower triagonal)");
    //Eigen::FullPivLU<Eigen::MatrixXd> solver;
    return solveLinearSystem<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::LDLT<Eigen::MatrixXd>>(
        Y, A, X, description + "(Y = A^-1 * X - dense + full matrix)");
}

template <class TYPE_A>
inline bool applyDynamicRegularization(Eigen::VectorXd& y, TYPE_A& A, const Eigen::VectorXd& x, const std::string& description, bool isLowerTriangular) {
    double dotProduct = y.dot(x);
    if (dotProduct <= 0.0 && x.squaredNorm() > 0.0) {
        Eigen::VectorXd stabRegularizer(A.rows());
        stabRegularizer.setZero();

        double currStabValue = 1e-5;
        for (int i = 0; i < 10; i++) {
            LENNY_LOG_DEBUG("%s: Applying dynamic regularization with stab value %lf", description.c_str(), currStabValue);

            stabRegularizer.setConstant(currStabValue);
            A += stabRegularizer.asDiagonal();

            solveLinearSystem(y, A, x, description, isLowerTriangular);

            dotProduct = y.dot(x);
            if (dotProduct > 0.0)
                return true;

            currStabValue *= 10.0;
        }
    }
    if (dotProduct < -1e-8)
        LENNY_LOG_WARNING("%s: Dynamic regularization was NOT successful -> dot product: %lf", description.c_str(), dotProduct);
    return false;
}

bool applyDynamicRegularization(Eigen::VectorXd& y, Eigen::SparseMatrixD& A, const Eigen::VectorXd& x, const std::string& description, bool isLowerTriangular) {
    return applyDynamicRegularization<Eigen::SparseMatrixD>(y, A, x, description, isLowerTriangular);
}

bool applyDynamicRegularization(Eigen::VectorXd& y, Eigen::MatrixXd& A, const Eigen::VectorXd& x, const std::string& description, bool isLowerTriangular) {
    return applyDynamicRegularization<Eigen::MatrixXd>(y, A, x, description, isLowerTriangular);
}

template <class TYPE_A>
void applyStaticRegularization(TYPE_A& A, const double& regWeight) {
    if (regWeight > 0.0) {
        Eigen::VectorXd stabRegularizer(A.rows());
        stabRegularizer.setZero();
        stabRegularizer.setConstant(regWeight);
        A += stabRegularizer.asDiagonal();
    }
}

void applyStaticRegularization(Eigen::SparseMatrixD& A, const double& regWeight) {
    applyStaticRegularization<Eigen::SparseMatrixD>(A, regWeight);
}

void applyStaticRegularization(Eigen::MatrixXd& A, const double& regWeight) {
    applyStaticRegularization<Eigen::MatrixXd>(A, regWeight);
}

bool applyInvertibilityCheck(const Eigen::SparseMatrixD& matrix, const std::string& description) {
    return applyInvertibilityCheck(matrix.toDense(), description);
}

bool applyInvertibilityCheck(const Eigen::MatrixXd& matrix, const std::string& description) {
    Eigen::FullPivLU<Eigen::MatrixXd> lu(matrix);
    if (!lu.isInvertible()) {
        LENNY_LOG_WARNING("%s -> matrix is NOT invertible", description.c_str());
        return false;
    }
    LENNY_LOG_INFO("%s -> matrix IS invertible!", description.c_str());
    return true;
}

bool applySymmetryCheck(const Eigen::SparseMatrixD& matrix, const std::string& description) {
    return applySymmetryCheck(matrix.toDense(), description);
}

bool applySymmetryCheck(const Eigen::MatrixXd& matrix, const std::string& description) {
    if (matrix.rows() != matrix.cols()) {
        LENNY_LOG_WARNING("%s -> matrix is NOT quadratic", description.c_str());
        return false;
    }
    const double norm = (matrix - matrix.transpose()).norm();
    if (norm > 1e-6) {
        LENNY_LOG_WARNING("%s -> matrix is NOT symmetric (norm: %lf)", description.c_str(), norm);
        return false;
    }
    LENNY_LOG_INFO("%s -> matrix IS symmetric", description.c_str());
    return true;
}

bool applyLowerTriangularCheck(const Eigen::SparseMatrixD& matrix, const std::string& description) {
    return applyLowerTriangularCheck(matrix.toDense(), description);
}

bool applyLowerTriangularCheck(const Eigen::MatrixXd& matrix, const std::string& description) {
    if (matrix.rows() != matrix.cols()) {
        LENNY_LOG_WARNING("%s -> matrix is NOT quadratic", description.c_str());
        return false;
    }
    if (!matrix.isLowerTriangular()) {
        LENNY_LOG_WARNING("%s -> matrix is NOT lower triangular", description.c_str());
        return false;
    }
    LENNY_LOG_INFO("%s -> matrix IS lower diagonal", description.c_str());
    return true;
}

inline bool evaluateLinearSystemSolveCheck(const double& testNorm, const std::string& description) {
    bool testPassed = false;
    if (IS_NAN(testNorm)) {
        LENNY_LOG_WARNING("%s -> testNorm is NaN (testNorm: %lf)", description.c_str(), testNorm);
    } else if (testNorm > 1e-6) {
        LENNY_LOG_WARNING("%s -> system is NOT solved correctly (testNorm: %lf)", description.c_str(), testNorm);
    } else {
        LENNY_LOG_INFO("%s -> system IS solved correctly (testNorm: %lf)", description.c_str(), testNorm);
        testPassed = true;
    }
    return testPassed;
}

template <class TYPE_Y, class TYPE_X>
inline bool applyLinearSystemSolveCheck(const TYPE_Y& y, const Eigen::MatrixXd& A, const TYPE_X& x, const std::string& description, bool isLowerTriangular) {
    Eigen::MatrixXd A_invertable;
    if (isLowerTriangular) {
        Eigen::MatrixXd D = A.diagonal().matrix().asDiagonal();
        A_invertable = A + A.transpose() - D;
    } else {
        A_invertable = A;
    }
    const TYPE_Y y_test = A_invertable.inverse() * x;
    const double testNorm = (y_test - y).norm();
    if (IS_NAN(testNorm)) {
        LENNY_LOG_WARNING("%s -> testNorm is NaN (testNorm: %lf)", description.c_str(), testNorm);
    } else if (testNorm > 1e-6) {
        LENNY_LOG_WARNING("%s -> system is NOT solved correctly (testNorm: %lf)", description.c_str(), testNorm);
    } else {
        LENNY_LOG_INFO("%s -> system IS solved correctly (testNorm: %lf)", description.c_str(), testNorm);
        return true;
    }
    return false;
}

bool applyLinearSystemSolveCheck(const Eigen::VectorXd& y, const Eigen::SparseMatrixD& A, const Eigen::VectorXd& x, const std::string& description,
                                 bool isLowerTriangular) {
    return applyLinearSystemSolveCheck<Eigen::VectorXd, Eigen::VectorXd>(y, A.toDense(), x, description, isLowerTriangular);
}

bool applyLinearSystemSolveCheck(const Eigen::VectorXd& y, const Eigen::MatrixXd& A, const Eigen::VectorXd& x, const std::string& description,
                                 bool isLowerTriangular) {
    return applyLinearSystemSolveCheck<Eigen::VectorXd, Eigen::VectorXd>(y, A, x, description, isLowerTriangular);
}

bool applyLinearSystemSolveCheck(const Eigen::MatrixXd& Y, const Eigen::SparseMatrixD& A, const Eigen::SparseMatrixD& X, const std::string& description,
                                 bool isLowerTriangular) {
    return applyLinearSystemSolveCheck<Eigen::MatrixXd, Eigen::SparseMatrixD>(Y, A.toDense(), X, description, isLowerTriangular);
}

bool applyLinearSystemSolveCheck(const Eigen::MatrixXd& Y, const Eigen::MatrixXd& A, const Eigen::MatrixXd& X, const std::string& description,
                                 bool isLowerTriangular) {
    return applyLinearSystemSolveCheck<Eigen::MatrixXd, Eigen::MatrixXd>(Y, A, X, description, isLowerTriangular);
}

}  // namespace lenny::optimization::utils