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
    std::vector<double> outputVector;
    double bias = this->Bias();

    for (int i=0; i < this->Rows(); i++) {
        // matrix multiplication weights*inputVector
        for (int j=0; j < this->Cols(); j++) {
            outputVector[i] += weights[i][j]*inputVector[j];
        }
        // add bias 
        outputVector[i] += bias;
    }

   return outputVector;
}