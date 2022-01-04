#include <Eigen/Dense>
#include <iostream>
#include "../src/loss_function.h"

void test_MSE() {
    auto mse = LossFunction("mse", 5);
    Eigen::VectorXd y{{1, 2, 2, 1, 0}};
    Eigen::VectorXd yLabel{{1, 1, 0, 1, -1}};

    std::cout << yLabel.size() << std::endl;

    std::cout << mse.ComputeLoss(y, yLabel) << std::endl;
    mse.UpdateGradient(y, yLabel);
    std::cout << mse.Gradient() << std::endl;
}

int main() {
    test_MSE();
}