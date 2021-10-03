#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_

#include <string>
#include <vector>

#include "activation.h"

class Perceptron;

class PerceptronData
/*
Class which determines the shape of the weight matrix of a perceptron
and its activation function.
*/
{
private:
    int _numberOfRows;
    int _numberOfCols;
    std::string _activation;

public:
    // (default) constructor
    PerceptronData(int rows=1, int cols=1, std::string="relu");

    // setters & getters
    void SetRows(int);
    int Rows() { return _numberOfRows; }
    void SetCols(int cols);
    int Cols() { return _numberOfCols; }
    void SetActivation(std::string activation)
    {
        _activation = activation;  // TODO: include check?!
    }
    std::string Activation() {
        return _activation;
    }
    Perceptron Initialize();
};


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
