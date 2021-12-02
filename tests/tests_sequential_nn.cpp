#include "../src/layer.h"
#include "../src/layer_cache.h"
#include "../src/transformation.h"
#include "../src/sequential_nn.h"
#include <iostream>
#include <vector>

void test_GetInitializationType() {
    auto ll_ptr = std::make_shared<LinearLayer>(2, 5);
    auto al_ptr = std::make_shared<ActivationLayer>(5, "tanh"); 

    std::cout << "Initialization type: " << SequentialNN::GetInitializationType(ll_ptr, al_ptr) << std::endl;
    ll_ptr->GetTransformation()->Initialize(SequentialNN::GetInitializationType(ll_ptr, al_ptr));
    std::cout << "Summary: " << ll_ptr->Summary() << std::endl;    
}

void test_ConnectLayers() {
    auto ll_ptr = std::make_shared<LinearLayer>(2, 5);
    auto al_ptr = std::make_shared<ActivationLayer>(2, "tanh"); 

    ll_ptr->GetLayerCache().ConnectForward(2, al_ptr->GetLayerCache());
    ll_ptr->Initialize("Xavier");
    std::vector<double> w({1, 2, 3, 4, 5});

    ll_ptr->Input(w);
    ll_ptr->Forward();

    std::cout << "Output of ll_ptr: " << Transformation::PrintDoubleVector(ll_ptr->Output()) << std::endl;
    std::cout << "Input of al_ptr: " << Transformation::PrintDoubleVector(*al_ptr->GetLayerCache().GetForwardInput()) << std::endl;

    al_ptr->Forward();
    std::cout << "Output of al_ptr: " << Transformation::PrintDoubleVector(al_ptr->Output()) << std::endl;
}

void test_SequentialNN() {
    auto ll_ptr = std::make_shared<LinearLayer>(2, 5);
    auto al_ptr = std::make_shared<ActivationLayer>(5, "relu"); 
    //std::vector<Layer> v{std::move(ll), std::move(al)};

    SequentialNN snn({ll_ptr, al_ptr});
    std::cout << snn.Summary() << std::endl;

    std::vector<double> w({1, 2, 3, 4, 5});

    snn.Initialize();
    std::cout << snn.Summary() << std::endl;

    snn.Input(w);
    snn.Forward();

    for (double a: snn.Output()) {
        std::cout << a << std::endl;
    }
}

int main() {
    test_ConnectLayers();
    //test_SequentialNN();
    //test_GetInitializationType();

    return 0;
}