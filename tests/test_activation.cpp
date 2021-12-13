#include "../src/activation.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>

void test_softmax() {
    auto softmax_activation = ActivationFct("softmax");
    std::vector<double> v{1, 2, 3, 4};
    std::vector<double> w = softmax_activation.Evaluate(v);
    double sum = 0.0;

    for (double d: w) {
        sum += d;
    }

    assert(sum == 1);
}

int main() {
    test_softmax();
    
    return 0;
}