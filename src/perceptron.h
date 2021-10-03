#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_

#include <string>
#include <vector>

#include "activation.h"
#include "perceptron_data.h"


class Perceptron
{
private:
    std::vector<std::vector<double> > _weights;
    double _bias;
    // perceptron data
    PerceptronData _data;
    // activation
    ActivationFct _activationFct;

public:
    // constructor
    // TODO #A: move to cpp-file?
    Perceptron(std::vector<std::vector<double> > weights, double bias, std::string fct)
        : _weights(weights), _bias(bias), _data(weights.size(), weights[0].size(), fct), 
        _activationFct(ActivationFct(fct))  
        
    {
    }

    // setters & getters
    void SetWeights(std::vector<std::vector<double> > weight_matrix) {
        _weights = weight_matrix;
    }
    std::vector< std::vector<double> > Weights() { return _weights; }
    void SetBias(double bias) { _bias = bias; }
    double Bias() { return _bias; }

    // simplify getters from data
    int Rows() { return _data.Rows(); }
    int Cols() { return _data.Cols(); }
    std::string Activation() { return _data.Activation(); }

    // setters & getters for activation function
    void SetActivationFct(std::string fct) {
        _activationFct = ActivationFct(fct);
    }
    ActivationFct GetActivationFct() { return _activationFct; }

    // evaluate on a vector
    std::vector<double> Evaluate(std::vector<double>);
};

#endif // PERCEPTRON_H_
