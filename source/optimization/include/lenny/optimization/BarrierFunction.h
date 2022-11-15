#pragma once

namespace lenny::optimization {

class BarrierFunction {
public:
    BarrierFunction(const double& stiffness, const double& epsilon);
    ~BarrierFunction() = default;

    //--- Setters
    void setStiffness(const double& stiffness);
    void setEpsilon(const double& epsilon);

    //--- Getters
    double getStiffness() const;
    double getEpsilon() const;

    //--- Members
    double computeValue(const double& x) const;
    double computeFirstDerivative(const double& x) const;
    double computeSecondDerivative(const double& x) const;

    //--- Gui
    void drawGui();

private:
    //--- Helpers
    void setup(const double& stiffness, const double& epsilon);

private:
    double a1, b1, c1, a2, b2, c2, d2, eps;
};

}  // namespace lenny::optimization