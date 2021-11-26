#include "../src/layer.h"
#include "../src/layer_cache.h"
#include "../src/transformation.h"
#include "../src/sequential_nn.h"
#include <iostream>
#include <vector>

void test_SequentialNN() {
    SequentialNN snn({LinearLayer(20, 50), ActivationLayer(20, "relu")});
}
