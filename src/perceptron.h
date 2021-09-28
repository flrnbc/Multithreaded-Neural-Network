#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_

#include <string>
#include <vector>

class PerceptronShape
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
    // constructor
    PerceptronShape(int, int, std::string);

    // setters
    void SetRows(int rows) { _numberOfRows = rows; }
    int GetRows() { return _numberOfRows; }
    void SetCols(int cols) { _numberOfCols = cols; }
    int GetCols() { return _numberOfCols; }
    void SetActivationFct(std::string activation) { _activationFct = activation; }
    // TODO: GetActivationFct when types are clear
};

class Perceptron
{
private:
    std::vector<std::vector<float>> _weights;
    float _bias;


}

#endif // PERCEPTRON_H_
