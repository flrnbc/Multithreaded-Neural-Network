#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#include <string>
#include <vector>

// collect activation functions
double heaviside(double);
double identity(double); 
double prelu(double, double);
double relu(double);
double sigmoid(double);
// tanh built-in


class ActivationFct{
private:
    std::string _name;
    double (*activation_fct)(double); // TODO #A: really need function pointer?

public:
    // default constructor
    ActivationFct();
    // constructor
    ActivationFct(std::string fct_name); // TODO #B: add possibility to add custom activation functions
    // methods
    std::vector<double> evaluate(std::vector<double>); // TODO #A: better pass by reference?
    std::string Name() {
        return _name; 
    }
};

#endif // ACTIVATION_H_
