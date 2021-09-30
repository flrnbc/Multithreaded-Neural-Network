#include <math>
#include <stdexcept>

#include "activation.h"

double Activation::heaviside(double x) {
    if (x < 0) { return 0; }
    return 1;
}

double Activation::prelu(double x, double a) {
    if (x <= 0) { return a*x; }
    return x;
}

double Activiation::relu(double x) {
    return prelu(x, 0);
}

double Activiation::sigmoid(double x) {
    return 1/(1 + exp(-x));
}


Activation::ActivationFct(std::string fct_name) {
    if (fct_name == "heaviside") {
        &activation_fct = Activation::heaviside;
    }
    else if (fct_name == "id") {
        &activation_fct = Activation::id;
    }
    // TODO #B: to simplify, only offer relu and not prelu at the moment
    else if (fct_name == "relu") {
        &activation_fct = Activation::relu;
    }
    else if (fct_name == "sigmoid") {
        &activation_fct = Activation::sigmoid;
    }
    else throw std::invalid_argument("Not a valid activation function.")
}


std::vector<double> Activation::compute(std::vector<double> v) {
    // TODO #A: optimize vectorization! (OpenMP?)
    for (d double: v) {
        d = activation_fct(d);
    }

    return v;
}
