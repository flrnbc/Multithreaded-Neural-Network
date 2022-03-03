#include "../src/layer_cache.h"
#include <Eigen/Dense>
#include <memory>
#include <iostream>

/**
 * Simple smoke tests for LayerCache class.
 */


void test_layer_cache() {   
    LayerCache lc1;
    LayerCache lc2;

    // test forward cache
    lc1.ConnectForward(5, 3, lc2);
    std::cout << "LayerCache lc2 forward_input: " << std::endl;
    std::cout << *lc2.GetForwardInput() << std::endl;
    std::cout << "LayerCache lc2 forward_input equals LayerCache lc1 forward_output: " << std::endl;
    std::cout << (*lc2.GetForwardInput() == *lc1.GetForwardOutput()) << std::endl;

    Eigen::MatrixXd m(5, 3);
    m << 1, 2, 3, 
         4, 156, 12, 
         33, 31, 342,
         234, 234, 634,
         1, 23, 4;

    *lc1.GetForwardOutput() = m;
    std::cout << "LayerCache lc2 forward_input (updated): " << std::endl;
    std::cout << *lc2.GetForwardInput() << std::endl;

    //lc1.SetBatchSize(10);
    std::cout << "LayerCache lc2 forward_input (updated): " << std::endl;
    std::cout << *lc2.GetForwardInput() << std::endl;

    // test backward cache
    lc1.ConnectBackward(8, 1, lc2);
    std::cout << "LayerCache lc1 backward_input: " << std::endl;
    std::cout << *lc1.GetBackwardInput() << std::endl;
    std::cout << "LayerCache lc1 backward_input equals LayerCache lc2 backward_output: " << std::endl;
    std::cout << (*lc1.GetBackwardInput() == *lc2.GetBackwardOutput()) << std::endl;

    Eigen::RowVectorXd w{{5, 6, 7, 8, 9, 10, 11, 12}};
    *lc2.GetBackwardOutput() = w;
    std::cout << "LayerCache lc1 backward_input (updated): " << std::endl;
    std::cout << *lc1.GetBackwardInput() << std::endl;
    std::cout << "LayerCache lc2 backward_output: " << std::endl;
    std::cout << *lc2.GetBackwardOutput() << std::endl;

    lc1.SetBatchSize(10);
    std::cout << "Forward again after changing batch size: " << std::endl;
    std::cout << *lc2.GetForwardInput() << std::endl;
}

// void test_layer_cache2() {
//     LayerCache lc1;
//     LayerCache lc2;

//     lc1.Connect(5, 8, lc2);

//     Eigen::VectorXd v{{5, 6, 7, 8, 9}};
//     Eigen::RowVectorXd w{{5, 6, 7, 8, 9, 10, 11, 12}};

//     *(lc1.GetForwardOutput()) = v;
//     *(lc2.GetBackwardOutput()) = w;

//     std::cout << "lc2 (forward input): \n" << *lc2.GetForwardInput() << std::endl;
//     std::cout << "lc1 (backward input): \n" << *lc1.GetBackwardInput() << std::endl;
// }

int main() {
    //test_layer_cache();
    test_layer_cache();

    return 0;
}