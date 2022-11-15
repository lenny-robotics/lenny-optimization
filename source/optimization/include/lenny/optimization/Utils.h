#pragma once

#include <lenny/tools/Definitions.h>

namespace lenny::optimization::utils {

//x = A*y -> y = A^-1*x
bool solveLinearSystem(Eigen::VectorXd& y, const Eigen::SparseMatrixD& A, const Eigen::VectorXd& x, const std::string& description, bool isLowerTriangular);
bool solveLinearSystem(Eigen::VectorXd& y, const Eigen::MatrixXd& A, const Eigen::VectorXd& x, const std::string& description, bool isLowerTriangular);
bool solveLinearSystem(Eigen::MatrixXd& Y, const Eigen::SparseMatrixD& A, const Eigen::SparseMatrixD& X, const std::string& description,
                       bool isLowerTriangular);
bool solveLinearSystem(Eigen::MatrixXd& Y, const Eigen::MatrixXd& A, const Eigen::MatrixXd& X, const std::string& description, bool isLowerTriangular);

bool applyDynamicRegularization(Eigen::VectorXd& y, Eigen::SparseMatrixD& A, const Eigen::VectorXd& x, const std::string& description, bool isLowerTriangular);
bool applyDynamicRegularization(Eigen::VectorXd& y, Eigen::MatrixXd& A, const Eigen::VectorXd& x, const std::string& description, bool isLowerTriangular);

void applyStaticRegularization(Eigen::SparseMatrixD& A, const double& regWeight);
void applyStaticRegularization(Eigen::MatrixXd& A, const double& regWeight);

bool applyInvertibilityCheck(const Eigen::SparseMatrixD& matrix, const std::string& description);
bool applyInvertibilityCheck(const Eigen::MatrixXd& matrix, const std::string& description);

bool applySymmetryCheck(const Eigen::SparseMatrixD& matrix, const std::string& description);
bool applySymmetryCheck(const Eigen::MatrixXd& matrix, const std::string& description);

bool applyLowerTriangularCheck(const Eigen::SparseMatrixD& matrix, const std::string& description);
bool applyLowerTriangularCheck(const Eigen::MatrixXd& matrix, const std::string& description);

bool applyLinearSystemSolveCheck(const Eigen::VectorXd& y, const Eigen::SparseMatrixD& A, const Eigen::VectorXd& x, const std::string& description,
                                 bool isLowerTriangular);
bool applyLinearSystemSolveCheck(const Eigen::VectorXd& y, const Eigen::MatrixXd& A, const Eigen::VectorXd& x, const std::string& description,
                                 bool isLowerTriangular);
bool applyLinearSystemSolveCheck(const Eigen::MatrixXd& Y, const Eigen::SparseMatrixD& A, const Eigen::SparseMatrixD& X, const std::string& description,
                                 bool isLowerTriangular);
bool applyLinearSystemSolveCheck(const Eigen::MatrixXd& Y, const Eigen::MatrixXd& A, const Eigen::MatrixXd& X, const std::string& description,
                                 bool isLowerTriangular);

}  // namespace lenny::optimization::utils