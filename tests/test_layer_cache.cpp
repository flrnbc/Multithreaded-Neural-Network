#include "../src/layer_cache.h"
#include <Eigen/Dense>
#include <memory>
#include <iostream>

void test_layer_cache() {   
    LayerCache lc1;
    LayerCache lc2;

    // test forward cache
    lc1.ConnectForward(5, lc2);
    std::cout << "LayerCache lc2 forward_input: " << std::endl;
    std::cout << *lc2.GetForwardInput() << std::endl;
    std::cout << "LayerCache lc2 forward_input equals LayerCache lc1 forward_output: " << std::endl;
    std::cout << (*lc2.GetForwardInput() == *lc1.GetForwardOutput()) << std::endl;

    Eigen::VectorXd v{{5, 6, 7, 8, 9}};
    *lc1.GetForwardOutput() = v;
    std::cout << "LayerCache lc2 forward_input (updated): " << std::endl;
    std::cout << *lc2.GetForwardInput() << std::endl;

    // test backward cache
    lc2.ConnectBackward(8, lc1);
    std::cout << "LayerCache lc1 backward_input: " << std::endl;
    std::cout << *lc1.GetBackwardInput() << std::endl;
    std::cout << "LayerCache lc1 backward_input equals LayerCache lc2 backward_output: " << std::endl;
    std::cout << (*lc1.GetBackwardInput() == *lc2.GetBackwardOutput()) << std::endl;

    Eigen::RowVectorXd w{{5, 6, 7, 8, 9, 10, 11, 12}};
    *lc2.GetBackwardOutput() = w;
    std::cout << "LayerCache lc1 backward_input (updated): " << std::endl;
    std::cout << *lc1.GetBackwardInput() << std::endl;
}

int main() {
    test_layer_cache();

    return 0;
}