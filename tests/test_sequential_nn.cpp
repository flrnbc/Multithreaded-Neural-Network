#include "../src/layer.h"
#include "../src/layer_cache.h"
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
    auto ll_ptr = std::make_shared<LinearLayer>(2, 5);
    auto al_ptr = std::make_shared<ActivationLayer>(2, "softmax"); 
    //std::vector<Layer> v{std::move(ll), std::move(al)};

    SequentialNN snn({ll_ptr, al_ptr});
    std::cout << snn.Summary() << std::endl;

    Eigen::VectorXd w{{1, 2, 3, 4, 5}};
    Eigen::RowVectorXd gradL{{1, 0}}; // typically gradient of loss function

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

int main() {
    //test_ConnectLayers();
    //test_SequentialNNForward();
    test_SequentialNNBackward();
    //test_GetInitializationType();

    return 0;
}