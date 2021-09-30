#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_

#include <string>
#include <vector>

#include "activation.h"

class PerceptronData
/*
Class which determines the shape of the weight matrix of a perceptron
and its activation function.
*/
{
private:
    int _numberOfRows;
    int _numberOfCols;
    std::string _activation; // TODO: might change to ActivationFct class

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
        _activation = activation;  // TODO: include check!
    }
    std::string Activation() {
        return _activation;
    }
};

class Perceptron
{
private:
    std::vector<std::vector<float>> _weights;
    float _bias;
    // perceptron data
    PerceptronData _data;
    // activation
    Activation::ActivationFct _activationFct;

public:
    // constructor
    Perceptron(std::vector<std::vector<float>> weights, float bias, std::string fct)
        : _weights(weights), _bias(bias), _activationFct Activation:ActivationFct(fct)
    {
    }

    // setters & getters
    void SetWeights(std::vector<std::vector<float>> weight_matrix) {
        _weights = weight_matrix;
    }
    std::vector<std::vector<float>> Weights() { return _weights; }
    void SetBias(float bias) { _bias = bias; }
    float Bias() { return _bias; }

    // simplify getters from data
    int Rows() { return _data.Rows(); }
    int Cols() { return _data.Cols(); }
    int Activation() { return _data.Activation(); }

    // setters & getters for activation function

    void SetActivationFct(std::string fct) {
        _activationFct = Activation:ActivationFct(fct);
    }
    Activation::ActivationFct ActivationFct() { return _activationFct; }
}

#endif // PERCEPTRON_H_
