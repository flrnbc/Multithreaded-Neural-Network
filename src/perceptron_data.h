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
    int _numberOfRows = 1;
    int _numberOfCols = 1;
    std::string _activation = "identity";

public:
    // default constructor
    PerceptronData() = default;

    // constructor
    PerceptronData(int rows, int cols, std::string);

    // setters & getters
    void SetRows(int);
    int Rows() { return _numberOfRows; }
    void SetCols(int);
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