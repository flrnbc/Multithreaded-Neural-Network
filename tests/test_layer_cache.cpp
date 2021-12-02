#include "../src/layer_cache.h"
#include <memory>
#include <iostream>

void test_layer_cache() {   
    LayerCache lc1;
    LayerCache lc2;

    lc1.ConnectForward(5, lc2);
    std::cout << (*lc2.GetForwardInput())[0] << std::endl;
    std::cout << (*lc2.GetForwardInput() == *lc1.GetForwardOutput()) << std::endl;

    std::vector<double> v{5, 6, 7, 8, 9};

    *lc1.GetForwardOutput() = v;
    std::cout << (*lc2.GetForwardInput())[0] << std::endl;
}

int main() {
    test_layer_cache();

    return 0;
}