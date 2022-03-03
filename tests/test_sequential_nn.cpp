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
    ll_ptr->GetLayerCache().Connect(2, 1, 5, al_ptr->GetLayerCache());

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

    // forward pass
    snn.Input(w);
    snn.Forward();
    std::cout << "Forward output: \n" << snn.Output() << std::endl;
    // summary
    std::cout << "With overloaded (): \n" << snn(w) << std::endl;

    // now matrix input
    Eigen::MatrixXd M(5, 3);
    M << 12, 5, 2,
         9, 2, 1, 
         5, 2, 5, 
         43, 2, 4,
         6, 2, 56;
    std::cout << "Transform matrix input: \n" << snn(M) << std::endl;
}

void test_SequentialNNBackward() {
    auto ll_ptr = std::make_shared<LinearLayer>(3, 5);
    auto al_ptr = std::make_shared<ActivationLayer>(3, "softmax"); 
    //std::vector<Layer> v{std::move(ll), std::move(al)};

    SequentialNN snn({ll_ptr, al_ptr});
    std::cout << snn.Summary() << std::endl;

    Eigen::MatrixXd input = Eigen::MatrixXd::Random(5, 4);
    Eigen::MatrixXd gradL = Eigen::MatrixXd::Random(4, 3); // {{1, 0, 0}}; // typically gradient of loss function

    snn.Initialize();
    std::cout << "Before training: \n" << snn.Summary() << std::endl;

    snn.Input(input);
    snn.Forward();
    //snn.Derivative();

    snn.BackwardInput(gradL);
    snn.Backward();
    std::cout << "Backward output: \n" << snn.BackwardOutput() << std::endl;

    // update weights
    snn.UpdateWeightsBias(0.1); 
    std::cout << "After training: \n" << snn.Summary() << std::endl;
}

void test_SequentialNNLoss() {
    auto ll_ptr = std::make_shared<LinearLayer>(5, 10);
    auto al_ptr = std::make_shared<ActivationLayer>(5, "softmax"); 
    auto mse = LossFunction("mse");
    //mse.SetCols(5);
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
    //snn.Derivative();

    std::cout << "Loss: " << mse(snn.Output(), wLabel) << std::endl;
}

void test_CrossEntropySoftmax() {
    SequentialNN snn({LinearLayer(5, 10), ActivationLayer(5, "softmax")});
    auto ce = LossFunction("cross_entropy");
    //ce.SetCols(5);

    Eigen::VectorXd w{{0, 1, 0, 2, 0, 3, 5, 10, 4, 100}};
    Eigen::VectorXd yLabel{{0, 1.0, 0, 0, 0}};

    snn.Initialize();
    std::cout << snn.Summary() << std::endl;

    snn.Input(w);
    snn.Forward();
    //snn.Derivative();
    std::cout << snn.Summary() << std::endl;
    std::cout << "Loss: " << ce(snn.Output(), yLabel) << std::endl;

    // try matrix input
    Eigen::MatrixXd M = Eigen::MatrixXd::Random(10, 4);
    std::cout << "Matrix output: \n" << snn(M) << std::endl;
    Eigen::MatrixXd Mlabel(5, 4);
    Mlabel << 0, 1.0, 0, 0,
              1.0, 0, 0, 0, 
              0, 0, 0, 1.0,
              1.0, 0, 0, 0,
              0, 1.0, 0, 0;

    //snn.Input(M);
    //snn.Forward();
    std::cout << "Loss: " << ce(snn.Output(), Mlabel) << std::endl;
}



int main() {
    //test_ConnectLayers();
    //test_SequentialNNForward();
    //test_SequentialNNBackward();
    //test_GetInitializationType();
    //test_SequentialNNLoss();
    //test_SequentialTest();
    
    //test_VectorInitialization();
    test_CrossEntropySoftmax();

    return 0;
}