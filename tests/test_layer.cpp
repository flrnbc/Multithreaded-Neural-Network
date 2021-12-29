
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

    ll.GetTransformation()->Initialize("Xavier"); 
    std::cout << ll.Summary() << std::endl;
    ll.Input(e1);
    ll.Forward();
    
    std::cout << "Transform e1: " << std::endl;
    for (int i=0; i<5; i++) {
        std::cout << ll.Output()[i] << std::endl;
    }

    //std::cout << ll.Summary() << std::endl;
}

void test_ActivationLayer() {
    ActivationLayer al(8, "relu");
    std::cout << al.Summary() << std::endl;

    Eigen::VectorXd v{{1, -5, 0, 1, 3, -6, -8, 0}};

    al.Input(v);
    al.Forward();

    std::cout << "Transform v: " << std::endl;
    for (int i=0; i<8; i++) {
        std::cout << al.Output()[i] << std::endl;
    }
}

int main() {
    test_LinearLayer();
    //test_ActivationLayer();
}