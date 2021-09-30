#ifndef ACTIVATION_H_
#define ACTIVATION_H_

// collect activation functions
double heaviside(double);
double id(double x) { return x; }
double prelu(double, double);
double relu(double);
double sigmoid(double);
// tanh built-in



#endif // ACTIVATION_H_
