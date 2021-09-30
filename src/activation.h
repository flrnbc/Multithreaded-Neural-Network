#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#include <string>
#include <vector>

// collect activation functions
double heaviside(double);
double id(double x) { return x; }
double prelu(double, double);
double relu(double);
double sigmoid(double);
// tanh built-in


class ActivationFct{
private:
    // do not allow changing the name or function pointer
    const std::string _name;
    const double (*activation_fct)(double);

public:
    // (default) constructor
    ActivationFct(std::string="relu"); // TODO #B: add possibility to add custom activation functions
    // methods
    std::vector<double> compute(std::vector<double>); // TODO #A: better pass by reference?
    std::string ActivationFct() {
        return name; // TODO #A: better to return the value pointed at?
    }
}

#endif // ACTIVATION_H_
