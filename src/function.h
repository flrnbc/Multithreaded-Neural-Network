#ifndef FUNCTION_H_
#define FUNCTION_H_

#include <cmath>
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
    double (*function)(double); // function pointer seems to be needed
    double (*derivative)(double); 

public:
    // constructor
    Function(std::string fct_name): 
        function(&identity),
        derivative(&identity_derivative)
        {
            // how to improve?
            if (fct_name == "identity") {
                function = &identity;
                derivative = &identity_derivative;
                }
            // TODO #B: to simplify, only offer relu and not prelu at the moment
            else if (fct_name == "relu") {
                function = &relu;
                derivative = &relu_derivative;
            }
            else if (fct_name == "sigmoid") {
                function = &sigmoid;
                derivative = &sigmoid_derivative;
            }
            // TODO: the following is just for convenience for implementing 
            // the actual softmax
            else if (fct_name == "softmax") { 
                function = &exp;
                derivative = &exp;
            }
            else if (fct_name == "tanh") {
                function = &tanh;
                derivative = &tanh_derivative;
            }
            else throw std::invalid_argument("Not a known function.");
            // finally set the name
            this->SetName(fct_name);
        }
    
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
