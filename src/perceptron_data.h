#ifndef PERCEPTRON_DATA_H_
#define PERCEPTRON_DATA_H_

#include <string>

#include "perceptron.h"

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
    PerceptronData(int rows=1, int cols=1, std::string="identity");

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



#endif // PERCEPTRON_DATA_H