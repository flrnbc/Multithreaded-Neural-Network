#include <cmath>
#include <stdexcept>
#include "function.h"

/************
 * FUNCTION *
 ************/

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
