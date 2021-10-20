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

Perceptron PerceptronData::Initialize() {
    // random initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    // 'quasi-linear' activation functions use different random initialization
    std::string quasiLinear [] = {"id", "sigmoid", "tanh"};
    // TODO #A: relu (and possibly prelu) now have the same initialization as heaviside which might not be ideal

    // perceptron data
    int cols = this->Cols();
    int rows = this->Rows();    
    std::vector<std::vector<double> > weights(rows, std::vector<double>(cols, 0));
    // TODO: we initialize the bias to a zero vector here which might not be optimal
    std::vector<double> bias(rows, 0);

    // initialize weights
    if (std::find(std::begin(quasiLinear), std::end(quasiLinear), this->Activation()) != std::end(quasiLinear)) {
        // use normalized Xavier weight initialization
        // TODO #A: give reference
        int inputPlusOutputSize = cols + rows;
        // NOTE: uniform_real_distribution(a, b) generates for [a, b) (half-open interval)
        std::uniform_real_distribution<> dis(-(sqrt(6)/sqrt(inputPlusOutputSize)), sqrt(6)/sqrt(inputPlusOutputSize));

        // randomly initialize weights
        // NOTE: be careful with cache-friendliness (outer loop over rows)
        // TODO #A: would be nicer to put this for-loop after the else-block (but then would have to declare dis before; but as what?)
        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {
                weights[i][j] = dis(gen); 
            }
        }
    } 
    else {
        // use He weight initialization
        // TODO #A: give reference
        std::normal_distribution<> dis(0.0, sqrt(2/cols));

        // randomly initialize weights
        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {
                weights[i][j] = dis(gen); 
            }
        }
    }
 
    return Perceptron(weights, bias, this->Activation());
}

// summary
std::string PerceptronData::Summary() {
    std::string rows = std::to_string(Rows());
    std::string cols = std::to_string(Cols());
    std::string size = "(" + rows + ", " + cols + ")";

    return "size: " + size + '\t' + " activation: " + Activation();
}