#include <Eigen/Dense>
#include <iostream>
#include "../src/loss_function.h"

/**
 * Smoke tests for LossFunction class.
 */

void test_MSE() {
    auto mse = LossFunction("mse");
    Eigen::MatrixXd y(2, 5); 
    y << 1, 2, 2, 1, 0,
         4, 0, 1, 2, 3;
    Eigen::MatrixXd yLabel(2, 5); 
    yLabel << 1, 1, 0, 1, -1,
              0, 1, -1, 0, 1;

    std::cout << "Total loss: " << mse(y, yLabel) << std::endl;
    mse.GradsAtPoints(y, yLabel);
    std::cout << mse.GetGrads() << std::endl;
}

// void test_MSE2() {
//     auto mse = LossFunction("mse");

//     Eigen::VectorXd y{{3}};
//     Eigen::VectorXd yLabel{{1}};

//     std::cout << mse(y, yLabel) << std::endl;

//     mse.GradsAtPoints(y, yLabel);
//     std::cout << "Gradient: " << mse.GetGrads() << std::endl;
// }

void test_cross_entropy() {
    auto ce = LossFunction("cross_entropy");

    Eigen::MatrixXd y(5, 3);
    Eigen::MatrixXd yLabel(5, 3);
    y << 0.2, 0.1, 0.7, 
         0.2, 0.0, 0.0,
         0.1, 0.6, 0.2,
         0.4, 0.0, 0.0,
         0.1, 0.3, 0.1;
    yLabel << 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 0.0,
              1.0, 0.0, 0.0,
              0.0, 0.0, 1.0;

    std::cout << ce(y, yLabel) << std::endl;
    ce.GradsAtPoints(y, yLabel);
    std::cout << ce.GetGrads() << std::endl;

    // change input size
    Eigen::MatrixXd y2 = Eigen::MatrixXd::Random(7, 4);
    Eigen::MatrixXd yLabel2 = Eigen::MatrixXd::Random(7, 4);

    std::cout << ce(y2, yLabel2) << std::endl;
    ce.GradsAtPoints(y2, yLabel2);
    std::cout << ce.GetGrads() << std::endl;
}


int main() {
    //test_MSE();
    //test_MSE2();
    test_cross_entropy();
}