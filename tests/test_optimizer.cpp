#include "../src/layer.h"
#include "../src/loss_function.h"
#include "../src/sequential_nn.h"
#include "../src/optimizer.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

void test_OptimizeLinearRegression() {
    auto ll_ptr = std::make_shared<LinearLayer>(1, 1);
    auto mse = LossFunction("mse");

    SequentialNN linear({ll_ptr});

    std::cout << linear.Summary() << std::endl;

    // assume y = a*x + c
    double a = 15.0;
    double c = 3.0;

    // define data points
    int data_points = 50;

    Eigen::MatrixXd X(1, data_points);
    Eigen::MatrixXd yLabel(1, data_points);

    for (int i=0; i<data_points; i++) {
        X(0, i) = i;
    }

    for (int i=0; i<data_points; i++) {
        yLabel(0, i) = a*X(0, i) + c;
    }

    std::cout << X << std::endl;

    // train with SDG optimizer
    auto sdg = SDG("mse", 0.001, 5000);

    sdg.Train(linear, X, yLabel);

    std::cout << linear.Summary() << std::endl;
}


int main() {
    test_OptimizeLinearRegression();
    return 0;
}