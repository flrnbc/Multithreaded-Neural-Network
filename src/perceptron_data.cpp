#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "activation.h"
#include "perceptron.h"
#include "perceptron_data.h"

void PerceptronData::SetRows(int rows) {
    if (rows < 1) throw std::invalid_argument("Not enough rows!"); // TODO #A: better handling of exceptions
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

// random numbers
// TODO: is there a performance issue? (since we create a generator etc. each time)?
double RandomNumberUniform(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);

    return dis(gen);
}

double RandomNumberNormal(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(min, max);

    return dis(gen);
}

std::vector<std::vector<double> > PerceptronData::WeightInitialization(int rows, int cols, std::string activationFct) {
    // 'quasi-linear' activation functions use different random initialization
    std::string quasiLinear [] = {"identity", "sigmoid", "tanh"};
    // TODO #A: relu (and possibly prelu) now have the same initialization as heaviside which might not be ideal
    // initialize weights to zeros
    std::vector<std::vector<double> > weights(rows, std::vector<double>(cols, 0));

    // actual initialization of weights
    if (std::find(std::begin(quasiLinear), std::end(quasiLinear), activationFct) != std::end(quasiLinear)) {      
        // use normalized Xavier weight initialization
        // TODO #A: give reference
        int inputPlusOutputSize = cols + rows;
        // NOTE: uniform_real_distribution(a, b) generates for [a, b) (half-open interval)
        double min = -(sqrt(6)/sqrt(inputPlusOutputSize));
        double max = sqrt(6)/sqrt(inputPlusOutputSize);

        // randomly initialize weights
        // NOTE: be careful with cache-friendliness (outer loop over rows)
        // TODO #A: would be nicer to put this for-loop after the else-block (but then would have to declare dis before; but as what?)
        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {
                weights[i][j] = RandomNumberUniform(min, max); 
            }
        }
    } 
    else {
        // use He weight initialization
        // TODO #A: give reference
        double min = 0.0;
        double max = sqrt(2.0/cols);

        // randomly initialize weights
        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {
                weights[i][j] = RandomNumberNormal(min, max); 
            }
        }
    }
    return weights;
}

// initialization of perceptron data to a perceptron
Perceptron PerceptronData::Initialize() {
    int rows = this->Rows();
    int cols = this->Cols();
    std::string activation = this->Activation();
    // TODO: we initialize the bias to a zero vector here which might not be optimal
    std::vector<double> bias(rows, 0);

    return Perceptron(PerceptronData::WeightInitialization(rows, cols, activation), bias, activation);
}

// summary
std::string PerceptronData::Summary() {
    std::string rows = std::to_string(Rows());
    std::string cols = std::to_string(Cols());
    std::string size = "(" + rows + ", " + cols + ")";

    return "size: " + size + '\t' + " activation: " + Activation();
}