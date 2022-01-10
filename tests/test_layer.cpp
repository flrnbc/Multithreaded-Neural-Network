
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>
#include "../src/transformation.h"
#include "../src/layer.h"
#include "../src/layer_cache.h"

void test_LinearLayer() {
    LinearLayer ll(5, 3);
    std::cout << ll.Summary() << std::endl;
    Eigen::VectorXd e1{{1, 0, 0}};
    Eigen::RowVectorXd w{{1, 0, 2, 0, 1}};

    ll.GetTransformation()->Initialize("Xavier"); 
    std::cout << ll.Summary() << std::endl;

    ll.Input(e1);
    ll.Forward();
    std::cout << "Transform e1: \n" << ll.Output() <<  std::endl;

    // test UpdateWeights
    ll.BackwardInput(w);
    ll.UpdateWeights();
    ll.UpdateBias();
    
    std::cout << "After updating weights and bias: " << ll.Summary() << std::endl;
}

void test_ActivationLayer() {
    ActivationLayer al(8, "softmax");
    std::cout << al.Summary() << std::endl;

    Eigen::VectorXd v{{1, -5, 0, 1, 3, -6, -8, 0}};
    Eigen::RowVectorXd w{{1, 0, 1, 0, 1, 0, 1, 0}};

    // forward pass
    al.Input(v);
    al.Forward();

    std::cout << "Transform v: \n" << al.Output() << std::endl;

    // update derivative
    al.UpdateDerivative();
    //al.GetTransformation()->UpdateDerivative(al.Output());
    std::cout << "After update: \n" << al.Summary() << std::endl;

    // backward pass
    al.BackwardInput(w);
    al.Backward();

    std::cout << "Backward output: \n" << al.BackwardOutput() << std::endl;
}

int main() {
    test_LinearLayer();
    //test_ActivationLayer();
}