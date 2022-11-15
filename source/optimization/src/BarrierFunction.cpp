#include <lenny/optimization/BarrierFunction.h>
#include <lenny/tools/Gui.h>

namespace lenny::optimization {

BarrierFunction::BarrierFunction(const double& stiffness, const double& epsilon) {
    setup(stiffness, epsilon);
}

void BarrierFunction::setup(const double& stiffness, const double& epsilon) {
    eps = epsilon;
    a1 = stiffness;
    b1 = 0.5 * a1 * epsilon;
    c1 = 1.0 / 6.0 * a1 * epsilon * epsilon;
    a2 = 1.0 / (2.0 * epsilon) * a1;
    b2 = a1;
    c2 = 0.5 * a1 * epsilon;
    d2 = 1.0 / 6.0 * a1 * epsilon * epsilon;
}

void BarrierFunction::setStiffness(const double& stiffness) {
    setup(stiffness, getEpsilon());
}

void BarrierFunction::setEpsilon(const double& epsilon) {
    setup(getStiffness(), epsilon);
}

double BarrierFunction::getStiffness() const {
    return a1;
}

double BarrierFunction::getEpsilon() const {
    return eps;
}

double BarrierFunction::computeValue(const double& x) const {
    if (x > 0.0)
        return 0.5 * a1 * x * x + b1 * x + c1;
    if (x > -eps)
        return 1.0 / 3.0 * a2 * x * x * x + 0.5 * b2 * x * x + c2 * x + d2;
    return 0.0;
}

double BarrierFunction::computeFirstDerivative(const double& x) const {
    if (x > 0.0)
        return a1 * x + b1;
    if (x > -eps)
        return a2 * x * x + b2 * x + c2;
    return 0.0;
}

double BarrierFunction::computeSecondDerivative(const double& x) const {
    if (x > 0.0)
        return a1;
    if (x > -eps)
        return 2.0 * a2 * x + b2;
    return 0.0;
}

void BarrierFunction::drawGui() {
    using tools::Gui;
    if (Gui::I->TreeNode("Barrier Function")) {
        double stiffness = getStiffness();
        if (Gui::I->Input("Stiffness", stiffness))
            setStiffness(stiffness);

        double epsilon = getEpsilon();
        if (Gui::I->Input("Epsilon", epsilon))
            setEpsilon(epsilon);

        Gui::I->TreePop();
    }
}

}  // namespace lenny::optimization