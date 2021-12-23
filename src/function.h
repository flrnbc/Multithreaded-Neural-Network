#ifndef FUNCTION_H_
#define FUNCTION_H_

#include <stdexcept>
#include <string>
#include <vector>

// collect functions
double identity(double);
double identity_derivative(double); 
double prelu(double, double);
double prelu_derivative(double);
double relu(double);
double relu_derivative(double);
double sigmoid(double);
double sigmoid_derivative(double);
// tanh built-in
double tanh_derivative(double);


class Function{
private:
    std::string _name;
    double (*function)(double) = nullptr; // function pointer seems to be needed
    double (*derivative)(double) = nullptr; 

public:
    // constructor
    Function(std::string fct_name="identity"); // TODO #B: add possibility to add custom activation functions

    // methods/operators
    double operator()(double input) {
        if (function == nullptr) {
            throw std::invalid_argument("Function not set.");
        }

        return (*function)(input);
    }
    
    double Derivative(double input) {
        if (derivative == nullptr) {
            throw std::invalid_argument("Derivative not set.");
        }

        return (*derivative)(input);
    }

    // setters/getters
    std::string Name() {
        return _name; 
    }
    void SetName(std::string fct_name) {
        _name = fct_name;
    }

};

#endif // FUNCTION_H_
