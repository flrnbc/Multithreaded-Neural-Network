#include <string>
#include <vector>

#include "activation.h"
#include "perceptron_data.h"
#include "perceptron.h"


// simplify getters from data
int Perceptron::Rows() { return (*_data).Rows(); }
int Perceptron::Cols() { return (*_data).Cols(); }
std::string Perceptron::Activation() { return (*_data).Activation(); }


Perceptron::Perceptron(std::vector<std::vector<double> > weights, double bias, std::string activation) {
    SetWeights(weights);
    SetBias(bias);
    // initialize perceptron data
    // TODO: test if it rejects 'empty vectors'
    (*_data) = PerceptronData(weights.size(), weights[0].size(), activation);
    // (*_data).SetActivation(activation);
    // (*_data).SetRows(weights.size());
    // (*_data).SetCols(weights[0].size());
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