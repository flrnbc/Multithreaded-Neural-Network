#include <math>

#include "activation.h"

double heaviside(double x) {
    if (x < 0) { return 0; }
    return 1;
}

double prelu(double x, double a) {
    if (x <= 0) { return a*x; }
    return x;
}

double relu(double x) {
    return prelu(x, 0);
}

double sigmoid(double x) {
    return 1/(1 + exp(-x));
}
