#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_

#include <string>
#include <vector>

class PerceptronData
/*
Class which determines the shape of the weight matrix of a perceptron
and its activation function.
*/
{
private:
    int _numberOfRows;
    int _numberOfCols;
    std::string _activationFct; // TODO: might change to ActivationFct class

public:
    // (default) constructor
    PerceptronData(int rows=1, int cols=1, std::string="ReLu");

    // setters & getters
    void SetRows(int);
    int Rows() { return _numberOfRows; }
    void SetCols(int cols);
    int Cols() { return _numberOfCols; }
    void SetActivationFct(std::string activation)
    {
        _activationFct = activation;  // TODO: include check!
    }
    // TODO: GetActivationFct when types are clear
};

class Perceptron
{
private:
    std::vector<std::vector<float>> _weights;
    float _bias;

public:
    // constructor
    Perceptron(std::vector<std::vector<float>>, float, std::string);

    // perceptron data
    PerceptronData data;

    // setters & getters
    void SetWeights(std::vector<std::vector<float>> weight_matrix) {
        _weights = weight_matrix;
    }
    std::vector<std::vector<float>> Weights() { return _weights; }
    void SetBias(float bias) { _bias = bias; }
    float Bias() { return _bias; }

    // simplify getters from data
    int Rows() { return data.Rows(); }
    int Cols() { return data.Cols(); }
    // TODO: get activiation funct when types are clear
}

#endif // PERCEPTRON_H_
