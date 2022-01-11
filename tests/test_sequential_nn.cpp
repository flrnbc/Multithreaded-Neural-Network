#include "../src/layer.h"
#include "../src/layer_cache.h"
#include "../src/loss_function.h"
#include "../src/transformation.h"
#include "../src/sequential_nn.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

void test_GetInitializationType() {
    auto ll_ptr = std::make_shared<LinearLayer>(2, 5);
    auto al_ptr = std::make_shared<ActivationLayer>(2, "tanh"); 

    std::cout << "Initialization type: " << SequentialNN::GetInitializationType(ll_ptr, al_ptr) << std::endl;
    ll_ptr->GetTransformation()->Initialize(SequentialNN::GetInitializationType(ll_ptr, al_ptr));
    std::cout << "Summary: " << ll_ptr->Summary() << std::endl;    
}

void test_ConnectLayers() {
    auto ll_ptr = std::make_shared<LinearLayer>(2, 5);
    auto al_ptr = std::make_shared<ActivationLayer>(2, "tanh"); 

    //ll_ptr->GetLayerCache().ConnectForward(2, al_ptr->GetLayerCache());
    ll_ptr->GetLayerCache().Connect(2, 5, al_ptr->GetLayerCache());

    ll_ptr->Initialize("Xavier");
    Eigen::VectorXd w{{1, 2, 3, 4, 5}};

    ll_ptr->Input(w);
    ll_ptr->Forward();

    std::cout << "Output of ll_ptr: " << ll_ptr->Output() << std::endl;
    std::cout << "Input of al_ptr: " << *al_ptr->GetLayerCache().GetForwardInput() << std::endl;

    al_ptr->Forward();
    std::cout << "Output of al_ptr: " << al_ptr->Output() << std::endl;
}

void test_SequentialNNForward() {
    auto ll_ptr = std::make_shared<LinearLayer>(2, 5);
    auto al_ptr = std::make_shared<ActivationLayer>(2, "relu"); 
    //std::vector<Layer> v{std::move(ll), std::move(al)};

    SequentialNN snn({ll_ptr, al_ptr});
    std::cout << snn.Summary() << std::endl;

    Eigen::VectorXd w{{1, 2, 3, 4, 5}};

    snn.Initialize();
    std::cout << snn.Summary() << std::endl;

    snn.Input(w);
    snn.Forward();

    std::cout << "Forward output: \n" << snn.Output() << std::endl;

    snn.UpdateDerivative();
    std::cout << "After update: " << snn.Summary() << std::endl;
}

void test_SequentialNNBackward() {
    auto ll_ptr = std::make_shared<LinearLayer>(3, 5);
    auto al_ptr = std::make_shared<ActivationLayer>(3, "softmax"); 
    //std::vector<Layer> v{std::move(ll), std::move(al)};

    SequentialNN snn({ll_ptr, al_ptr});
    std::cout << snn.Summary() << std::endl;

    Eigen::VectorXd w{{1, 2, 3, 4, 5}};
    Eigen::RowVectorXd gradL{{1, 0, 0}}; // typically gradient of loss function

    snn.Initialize();
    std::cout << snn.Summary() << std::endl;

    snn.Input(w);
    snn.Forward();
    snn.UpdateDerivative();

    snn.BackwardInput(gradL);
    snn.Backward();

    std::cout << snn.Summary() << std::endl;

    std::cout << "Backward output: \n" << snn.BackwardOutput() << std::endl;
}

void test_SequentialNNLoss() {
    auto ll_ptr = std::make_shared<LinearLayer>(5, 10);
    auto al_ptr = std::make_shared<ActivationLayer>(5, "softmax"); 
    auto mse = LossFunction("mse", 5);
    //std::vector<Layer> v{std::move(ll), std::move(al)};

    SequentialNN snn({ll_ptr, al_ptr});
    std::cout << snn.Summary() << std::endl;

    Eigen::VectorXd w{{1, 2, 3, 4, 5, 6, 2, 3, 6, 9}};
    Eigen::VectorXd wLabel{{0, 0.5, 0.6, 0.3, 1.2}};
    Eigen::VectorXd w1{{2, 6, 3, 1, 4, 6, 2, 7, 3, 1}};
    Eigen::VectorXd w1Label{{1.4, 5.2, 2.1, 1.8, 2.9}};

    snn.Initialize();
    std::cout << snn.Summary() << std::endl;

    snn.Input(w);
    snn.Forward();
    snn.UpdateDerivative();

    std::cout << "Loss: " << snn.Loss(mse, wLabel) << std::endl;

    snn.UpdateBackwardInput(mse, wLabel);
    snn.Backward();

    std::cout << snn.Summary() << std::endl;
    std::cout << "Backward output: \n" << snn.BackwardOutput() << std::endl;
}

void test_SequentialTest() {
    auto ll_ptr = std::make_shared<LinearLayer>(5, 10);
    auto al_ptr = std::make_shared<ActivationLayer>(5, "softmax"); 
    auto mse = LossFunction("mse", 5);
    //std::vector<Layer> v{std::move(ll), std::move(al)};

    SequentialNN snn({ll_ptr, al_ptr});

    Eigen::VectorXd w{{1, 2, 3, 4, 5, 6, 2, 3, 6, 9}};
    Eigen::VectorXd wLabel{{0, 0.5, 0.6, 0.3, 1.2}};

    //snn.Initialize();
    std::cout << "Before training: " << snn.Summary() << std::endl;

    snn.Train(mse, 0.0001, w, wLabel);
    
    std::cout << "After training: " << snn.Summary() << std::endl;
}

void test_TrainLinearRegression() {
    auto ll_ptr = std::make_shared<LinearLayer>(1, 1);
    auto mse = LossFunction("mse", 1);

    SequentialNN linear({ll_ptr});

    std::cout << linear.Summary() << std::endl;

    // assume y = a*x + c
    double a = 15.0;
    double c = 3.0;

    // 
    int data_points = 20;
    

    Eigen::MatrixXd X(1, data_points);
    Eigen::MatrixXd yLabel(1, data_points);

    for (int i=0; i<data_points; i++) {
        X(0, i) = i;
    }

    for (int i=0; i<data_points; i++) {
        yLabel(0, i) = a*X(0, i) + c;
    }

    std::cout << X << std::endl;

    for (int j=0; j<500; j++) {
        std::cout << "----------- Epoch " << j << " -----------" << std::endl;
        for (int i=0; i<data_points; i++) {
            linear.Train(mse, 0.001, X.col(i), yLabel.col(i));
        }
    }

    std::cout << linear.Summary() << std::endl;

    // by hand
    // double w=0;
    // double b=0;
    // double x=0;
    // double y=0;

    // for (int j=0; j<1000; j++) {
    //     for (int i=0; i<50; i++) {
    //         x = X(0, i);
    //         y = w*x+b;
    //         w += -0.001*2*(y - yLabel(0, i))*x;
    //         b += -0.001*2*(y - yLabel(0, i));
    //         std::cout << w << "\t" << b << std::endl;
    //     }
    // }

    // std::cout << "By hand: \n" << "weight: " << w << "\nbias: " << b << std::endl;
}

int main() {
    //test_ConnectLayers();
    //test_SequentialNNForward();
    //test_SequentialNNBackward();
    //test_GetInitializationType();
    //test_SequentialNNLoss();
    //test_SequentialTest();
    test_TrainLinearRegression();

    return 0;
}