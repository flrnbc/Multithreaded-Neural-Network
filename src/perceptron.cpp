#include <stdexcept>
#include <string>

#include "perceptron.h"


void PerceptronData::SetRows(int rows) {
    if (rows < 1) throw std::invalid_argument("Not enough rows!");
    _numberOfRows = rows;
}


void PerceptronData::SetCols(int cols) {
    if (cols < 1) throw std::invalid_argument("Not enough cols!");
    _numberOfCols = cols;
}


PerceptronData::PerceptronData(int rows, int cols, std::string activation) {
    if (rows < 1 or cols < 1) throw std::invalid_argument("Not enough rows or columns!");
    SetRows(rows);
    SetCols(cols);
    // TODO: add check for activation function
    SetActivation(activation);
}


Perceptron::Perceptron(std::vector<std::vector<float>> weights, float bias, std::string activation) {
    SetWeights(weights);
    SetBias(bias);
    // initialize perceptron data
    // TODO: test if it rejects 'empty vectors'
    _data.SetActivation(activation);
    _data.SetRows(weights.size());
    _data.SetCols(weights[0].size());
}
