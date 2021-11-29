#include "../src/layer.h"
#include "../src/layer_cache.h"
#include "../src/transformation.h"
#include "../src/sequential_nn.h"
#include <iostream>
#include <vector>

void test_SequentialNN() {
    LinearLayer ll(20, 50);
    ActivationLayer al(20, "relu"); 
    std::vector<std::shared_ptr<Layer> > v{std::make_shared<LinearLayer>(ll), std::make_shared<ActivationLayer>(al)};

    SequentialNN snn(v);
    std::vector<double> w(50, 1);

    snn.Input(w);
    snn.Forward();
}
