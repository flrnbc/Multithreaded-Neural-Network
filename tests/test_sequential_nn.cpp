#include "../src/layer.h"
#include "../src/loss_function.h"
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

void test_VectorInitialization() {
    std::vector<Layer> vlayers{{LinearLayer(4, 8), 
                                ActivationLayer(4, "relu"), 
                                LinearLayer(2, 4),
                                ActivationLayer(2, "softmax")
                                }};
    SequentialNN snn(vlayers);

    std::cout << snn.Summary() << std::endl;
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

    std::cout << "With overloaded (): \n" << snn(w) << std::endl;
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
    auto mse = LossFunction("mse");
    mse.SetCols(5);
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

    std::cout << "Loss: " << mse(snn.Output(), wLabel) << std::endl;
}



int main() {
    //test_ConnectLayers();
    test_SequentialNNForward();
    //test_SequentialNNBackward();
    //test_GetInitializationType();
    //test_SequentialNNLoss();
    //test_SequentialTest();
    
    //test_VectorInitialization();

    return 0;
}