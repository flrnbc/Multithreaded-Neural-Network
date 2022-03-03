#include "../src/layer.h"
#include "../src/loss_function.h"
#include "../src/sequential_nn.h"
#include "../src/optimizer.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

/**
 * Tests for Optimizer class.
 */

void test_Step() {
    // Testing a training step of Optimizer to check the basic functionality.
    auto ll_ptr = std::make_shared<LinearLayer>(5, 10);
    auto al_ptr = std::make_shared<ActivationLayer>(5, "softmax"); 
    auto mse = LossFunction("mse");
    SequentialNN snn({ll_ptr, al_ptr});
    auto sdg = SDG("mse", 3, 0.1);

    Eigen::VectorXd w{{1, 2, 3, 4, 5, 6, 2, 3, 6, 9}};
    Eigen::VectorXd wLabel{{0, 0.5, 0.6, 0.3, 1.2}};

    // single training step
    std::cout << "Before training step: " << snn.Summary() << std::endl;
    sdg.Step(snn, w, wLabel);
    std::cout << "After training step: " << snn.Summary() << std::endl;
}

void test_TrainLinearRegression() {
    // Training test with the simplest linear model: 1-dimensional linear regression (affine-linear transformation)
    // determined by two doubles (a*x + b).
    // Compare with 'training by hand', i.e. manually implementing gradient descent. 

    auto ll_ptr = std::make_shared<LinearLayer>(1, 1);
    auto mse = LossFunction("mse");
    auto sdg = SDG("mse", 5, 0.001);
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



void test_OptimizeLinearRegression1D() {
    // streamlined training of 1-dimensional linear regression
    auto ll_ptr = std::make_shared<LinearLayer>(1, 1);
    auto act_ptr = std::make_shared<ActivationLayer>(1, "identity");
    auto mse = LossFunction("mse");

    SequentialNN linear({ll_ptr, act_ptr});

    std::cout << linear.Summary() << std::endl;

    // assume y = a*x + c
    // using vectors better adapted for higher-dimensional examples
    Eigen::VectorXd a(1);
    a << 15.0;
    Eigen::VectorXd c(1);
    c << 3.0;

    // define data points
    int data_points = 50;

    Eigen::MatrixXd X(1, data_points);
    Eigen::MatrixXd yLabel(1, data_points);

    for (int i=0; i<data_points; i++) {
        X(0, i) = i;
    }
    for (int i=0; i<data_points; i++) {
        yLabel(0, i) = (act_ptr->GetTransformation()->Transform(a*X.col(i) + c))(0);
    }
    std::cout << yLabel << std::endl;

    // train with SDG optimizer
    auto sdg = SDG("mse", 1, 0.0001);
    sdg.Train(linear, X, yLabel, 30000);
    std::cout << linear.Summary() << std::endl;
}

void test_OptimizeLinearRegression2D() {
    // train 2-dimensional linear regression
    auto ll_ptr = std::make_shared<LinearLayer>(2, 2);
    auto act_ptr = std::make_shared<ActivationLayer>(2, "softmax");
    SequentialNN linear({ll_ptr, act_ptr});
    std::cout << linear.Summary() << std::endl;

    // assume y = A*x + c
    Eigen::MatrixXd A(2, 2);
    Eigen::VectorXd c(2);
    A << 1.0, 2.0,
         0.0, 5.0;
    c << 1.0, 8.0;
    
    // define data points
    int data_points = 5000;
    Eigen::MatrixXd X(2, data_points);
    Eigen::MatrixXd yLabel(2, data_points);

    for (int i=0; i<data_points; i++) {
        X(0, i) = i;
        X(1, i) = 10*i;
    }
    for (int i=0; i<data_points; i++) {
        yLabel.col(i) = A*X.col(i) + c;
    }
    std::cout << X << std::endl;

    // train with SDG optimizer
    auto sdg = SDG("mse", 5, 1e-9);
    sdg.Train(linear, X, yLabel, 5000);
    std::cout << linear.Summary() << std::endl;
}

void test_Softmax() {
    SequentialNN snn({LinearLayer(3, 6), ActivationLayer(3, "softmax")});

    int numberOfSamples = 1000;

    Eigen::MatrixXd X(6, 4000);
    Eigen::MatrixXd yLabel(3, 4000);
    Eigen::MatrixXd Xtest(6, 1000);
    Eigen::MatrixXd yTest(3, 1000);

    std::cout << snn.Summary() << std::endl;

    // construct samples
    for (int i=0; i<4000; i++) {
        X.col(i) = Eigen::VectorXd::LinSpaced(6, i, i*i); // very simple choice...
    }
    for (int i=0; i<1000; i++) {
        Xtest.col(i) = Eigen::VectorXd::LinSpaced(6, i+4000, i+4018);
    }
    // construct labels/output somewhat strangely...
    for (int j=0; j<4000; j++) {
        if (j % 8 == 0) {
            yLabel(0, j) = 1;
            yLabel(1, j) = 0;
        } else{
            yLabel(0, j) = 0;
            yLabel(1, j) = 1;
        }
    }
    for (int j=0; j<1000; j++) {
        if (j % 4 == 0) {
            yTest(0, j) = 1;
            yTest(1, j) = 0;
        }
        yTest(0, j) = 0;
        yTest(1, j) = 1;
        
    }
    //std::cout << "X: " << X << std::endl;
    //std::cout << "yLabel: " << yLabel << std::endl;
    // train
    auto sdg = SDG("cross_entropy", 5, 0.1);
    sdg.Train(snn, X, yLabel, 2000);
    std::cout << snn.Summary() << std::endl;
}

int main() {
    //test_OptimizeLinearRegression2D();
    //test_OptimizeLinearRegression1D();
    //test_Step();
    test_Softmax();
    return 0;
}