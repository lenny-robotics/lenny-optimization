#pragma once

#include <gtest/gtest.h>
#include <lenny/optimization/EqualityConstraint.h>
#include <lenny/optimization/InequalityConstraint.h>
#include <lenny/optimization/NewtonOptimizer.h>
#include <lenny/optimization/TotalObjective.h>

#include <iostream>

namespace test {

class TestObjective : public lenny::optimization::Objective {
public:
    TestObjective() : lenny::optimization::Objective("TestObjective") {}

    double computeValue(const Eigen::VectorXd& x) const override {
        double value = 0.0;
        for (int i = 0; i < x.size(); i++)
            value += 0.5 * x[i] * x[i];
        return value;
    }

    void computeGradient(Eigen::VectorXd& pVpX, const Eigen::VectorXd& x) const override {
        //        estimateGradient(pVpX, x);
        pVpX.resize(x.size());
        for (int i = 0; i < x.size(); i++)
            pVpX[i] = x[i];
    }

    void computeHessian(Eigen::SparseMatrixD& p2VpX2, const Eigen::VectorXd& x) const override {
        //        estimateHessian(p2VpX2, x);
        p2VpX2.resize(x.size(), x.size());
        p2VpX2.setIdentity();
    }
};

class TestEqualityConstraint : public lenny::optimization::EqualityConstraint {
public:
    TestEqualityConstraint() : lenny::optimization::EqualityConstraint("TestEqualityConstraint") {}

    uint getConstraintNumber() const override {
        return 1;
    }

    void computeConstraint(Eigen::VectorXd& C, const Eigen::VectorXd& x) const {
        C.resize(getConstraintNumber());
        C[0] = 0.5 * (x[0] - 1.0) * (x[0] - 1.0);
    }

    void computeJacobian(Eigen::SparseMatrixD& pCpX, const Eigen::VectorXd& x) const {
        //estimateJacobian(pCpX, x);
        pCpX.resize(getConstraintNumber(), x.size());
        pCpX.setZero();
        pCpX.coeffRef(0, 0) = (x[0] - 1.0);
    }

    void computeTensor(Eigen::TensorD& p2CpX2, const Eigen::VectorXd& x) const {
        //estimateTensor(p2CpX2, x);
        p2CpX2.resize(Eigen::Vector3i(getConstraintNumber(), x.size(), x.size()));
        p2CpX2.addEntry(Eigen::Vector3i(0, 0, 0), 1.0);
    }
};

class TestInequalityConstraint : public lenny::optimization::InequalityConstraint {
public:
    TestInequalityConstraint() : lenny::optimization::InequalityConstraint("TestInequalityConstraint") {}

    uint getConstraintNumber() const override {
        return 1;
    }

    void computeConstraint(Eigen::VectorXd& C, const Eigen::VectorXd& x) const {
        C.resize(getConstraintNumber());
        C[0] = 0.5 * (x[1] - 1.0) * (x[1] - 1.0);
    }

    void computeJacobian(Eigen::SparseMatrixD& pCpX, const Eigen::VectorXd& x) const {
        pCpX.resize(getConstraintNumber(), x.size());
        pCpX.setZero();
        pCpX.coeffRef(0, 1) = (x[1] - 1.0);
    }

    void computeTensor(Eigen::TensorD& p2CpX2, const Eigen::VectorXd& x) const {
        p2CpX2.resize(Eigen::Vector3i(getConstraintNumber(), x.size(), x.size()));
        p2CpX2.addEntry(Eigen::Vector3i(0, 1, 1), 1.0);
    }
};

class TotalTestObjective : public lenny::optimization::TotalObjective {
public:
    TotalTestObjective() : lenny::optimization::TotalObjective("TotalTestObjective") {
        subObjectives.push_back({std::make_unique<TestObjective>(), 1.25});
        subObjectives.push_back({std::make_unique<TestEqualityConstraint>(), 10.0});
        subObjectives.push_back({std::make_unique<TestInequalityConstraint>(), 10.0});
    }
};

}  // namespace test

TEST(optimization, test) {
    Eigen::VectorXd x = Eigen::VectorXd::Random(2);
    test::TotalTestObjective totalObjective;
    lenny::optimization::NewtonOptimizer optimizer("TestOptimizer");

    EXPECT_TRUE(totalObjective.testIndividualFirstDerivatives(x));
    EXPECT_TRUE(totalObjective.testIndividualSecondDerivatives(x));

    EXPECT_TRUE(totalObjective.testGradient(x));
    EXPECT_TRUE(totalObjective.testHessian(x));

    EXPECT_FALSE(totalObjective.checkConstraintSatisfaction(x));

    optimizer.checkHessianProperties = true;
    optimizer.checkLinearSystemSolve = true;
    //optimizer.useSparseSolver = false;

    std::cout << "START - x: " << x.transpose() << std::endl;
    for (int i = 0; i < 10; i++)
        optimizer.optimize(x, totalObjective, 1);
    std::cout << "END - x: " << x.transpose() << std::endl;
}