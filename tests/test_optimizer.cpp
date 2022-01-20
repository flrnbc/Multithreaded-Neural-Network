#include "../src/layer.h"
#include "../src/loss_function.h"
#include "../src/sequential_nn.h"
#include "../src/optimizer.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

void test_Step() {
    auto ll_ptr = std::make_shared<LinearLayer>(5, 10);
    auto al_ptr = std::make_shared<ActivationLayer>(5, "softmax"); 
    auto mse = LossFunction("mse");
    SequentialNN snn({ll_ptr, al_ptr});
    auto sdg = SDG("mse", 0.1);

    Eigen::VectorXd w{{1, 2, 3, 4, 5, 6, 2, 3, 6, 9}};
    Eigen::VectorXd wLabel{{0, 0.5, 0.6, 0.3, 1.2}};

    // single training step
    std::cout << "Before training step: " << snn.Summary() << std::endl;
    sdg.Step(snn, w, wLabel);
    std::cout << "After training step: " << snn.Summary() << std::endl;
}

void test_TrainLinearRegression() {
    auto ll_ptr = std::make_shared<LinearLayer>(1, 1);
    auto mse = LossFunction("mse");
    auto sdg = SDG("mse", 0.001);
    SequentialNN linear({ll_ptr});

    std::cout << linear.Summary() << std::endl;

    // assume y = a*x + c
    double a = 15.0;
    double c = 3.0;

    // number of generated data points
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

    for (int j=0; j<100; j++) {
        std::cout << "----------- Epoch " << j << " -----------" << std::endl;
        for (int i=0; i<data_points; i++) {
            sdg.Step(linear, X.col(i), yLabel.col(i));
        }
    }

    //by hand
    double w=0;
    double b=0;
    double x=0;
    double y=0;

    for (int j=0; j<100; j++) {
        for (int i=0; i<50; i++) {
            x = X(0, i);
            y = w*x+b;
            w += -0.001*2*(y - yLabel(0, i))*x;
            b += -0.001*2*(y - yLabel(0, i));
            //std::cout << w << "\t" << b << std::endl;
        }
    }

    std::cout << "By hand: \n" << "weight: " << w << "\nbias: " << b << std::endl;
    std::cout << "With our implementation: \n" << linear.Summary() << std::endl;
}



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
    auto sdg = SDG("mse", 0.00001);
    sdg.Train(linear, X, yLabel, 50000);
    std::cout << linear.Summary() << std::endl;
}


int main() {
    //test_OptimizeLinearRegression();
    //test_TrainLinearRegression();
    test_Step();
    return 0;
}