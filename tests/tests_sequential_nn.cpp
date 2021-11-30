#include "../src/layer.h"
#include "../src/layer_cache.h"
#include "../src/transformation.h"
#include "../src/sequential_nn.h"
#include <iostream>
#include <vector>

void test_SequentialNN() {
    auto ll_ptr = std::make_shared<LinearLayer>(2, 5);
    auto al_ptr = std::make_shared<ActivationLayer>(5, "relu"); 
    //std::vector<Layer> v{std::move(ll), std::move(al)};

    SequentialNN snn({ll_ptr, al_ptr});
    std::cout << snn.Summary() << std::endl;

    std::vector<double> w(5, 1);

    snn.Input(w);
    snn.Forward();

    for (double a: snn.Output()) {
        std::cout << a << std::endl;
    }
}

int main() {
    test_SequentialNN();

    return 0;
}