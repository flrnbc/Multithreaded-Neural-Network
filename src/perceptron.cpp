#include <string>
#include <vector>

#include "activation.h"
#include "perceptron_data.h"
#include "perceptron.h"


Perceptron::Perceptron(std::vector<std::vector<double> > weights, double bias, std::string activation) {
    SetWeights(weights);
    SetBias(bias);
    // initialize perceptron data
    // TODO: test if it rejects 'empty vectors'
    SetRows(weights.size());
    SetCols(weights[0].size());
    SetActivationFct(activation);

    // NOTE: using PerceptronData did not work...
    //(*_data) = PerceptronData(weights.size(), weights[0].size(), activation);
}


std::vector<double> Perceptron::Evaluate(std::vector<double> inputVector) {
    std::vector<std::vector<double> > weights = this->Weights();
    std::vector<double> outputVector(weights.size(), 0);
    double bias = this->Bias();
    int rows = this->Rows();
    int cols = this->Cols();

    for (int i=0; i < rows; i++) {
        // matrix multiplication weights*inputVector
        for (int j=0; j < cols; j++) {
            outputVector[i] += weights[i][j]*inputVector[j];
        }
        // add bias 
        outputVector[i] += bias;
    }

   return outputVector;
}