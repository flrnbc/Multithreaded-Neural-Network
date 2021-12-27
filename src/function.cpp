#include <cmath>
#include <stdexcept>

#include "function.h"

// Functions which will be used
double identity(double x) {
    return x;
}

double identity_derivative(double x) {
    return 1.0;
}

double prelu(double x, double a) {
    if (x <= 0) { return a*x; }
    return x;
}

double prelu_derivative(double x, double a) {
    if (x <= 0) { return a; }
    return 1;
}

double relu(double x) {
    return prelu(x, 0);
}

double relu_derivative(double x) {
    return prelu_derivative(x, 1.0);
}

double sigmoid(double x) {
    return 1/(1 + exp(-x));
}

double sigmoid_derivative(double x) {
    return sigmoid(x)*(1- sigmoid(x));
}

double tanh_derivative(double x) {
    return (1 - tanh(x)*tanh(x));
}


/* std::vector<double> Function::Evaluate(std::vector<double> v) {
    // TODO #A: optimize vectorization?! (OpenMP?)
    for (double& d: v) {
        d = function(d);
    }

    // TODO: that's not ideal but ok for now...
    if (_name == "softmax") {
        double sum = 0.0;

        for (double& d: v) {
            sum += d; // note the definition of softmax above
        }

        for (double& d: v) {
            d /= sum; // sum != 0
        }
    }

    return v;
} */
