#include "../src/function.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>

// void test_softmax() {
//     auto softmax_activation = ActivationFct("softmax");
//     std::vector<double> v{1, 2, 3, 4};
//     std::vector<double> w = softmax_activation.Evaluate(v);
//     double sum = 0.0;

//     for (double d: w) {
//         sum += d;
//     }

//     assert(sum == 1);
// }

void test_functions() {
    auto f1 = Function("identity");
    auto f2 = Function("sigmoid");
    auto f3 = Function("tanh");
    std::vector<Function> functions{f1, f2, f3};

    double a = 500.0;

    for (Function& fct: functions) {
        std::cout << fct(a) << std::endl;
        std::cout << fct.Derivative(a) << std::endl;
    }
}

int main() {
    test_functions();
    
    return 0;
}