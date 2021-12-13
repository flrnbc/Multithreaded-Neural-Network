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
double softmax(double);
// tanh built-in


class ActivationFct{
private:
    std::string _name;
    double (*activation_fct)(double); // function pointer seems to be needed

public:
    // (default) constructor
    ActivationFct(std::string fct_name="identity"); // TODO #B: add possibility to add custom activation functions
    // methods
    std::vector<double> Evaluate(std::vector<double>); // want to keep input, so no pass by reference
    // setters/getters
    std::string Name() {
        return _name; 
    }
    void SetName(std::string fct_name) {
        _name = fct_name;
    }
};

#endif // ACTIVATION_H_
