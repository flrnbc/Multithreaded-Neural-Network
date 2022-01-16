#include <Eigen/Dense>
#include <iostream>
#include "../src/loss_function.h"

void test_MSE() {
    auto mse = LossFunction("mse");
    mse.SetCols(5);
    Eigen::VectorXd y{{1, 2, 2, 1, 0}};
    Eigen::VectorXd yLabel{{1, 1, 0, 1, -1}};

    std::cout << yLabel.size() << std::endl;

    std::cout << mse.ComputeLoss(y, yLabel) << std::endl;
    mse.UpdateGradient(y, yLabel);
    std::cout << mse.Gradient() << std::endl;
}

void test_MSE2() {
    auto mse = LossFunction("mse");
    mse.SetCols(1);

    Eigen::VectorXd y{{3}};
    Eigen::VectorXd yLabel{{1}};

    std::cout << mse.ComputeLoss(y, yLabel) << std::endl;

    mse.UpdateGradient(y, yLabel);
    std::cout << "Gradient: " << mse.Gradient() << std::endl;
}

int main() {
    //test_MSE();
    test_MSE2();
}