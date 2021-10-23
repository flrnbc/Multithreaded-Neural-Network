#include <string>
#include <vector>

#include "activation.h"
#include "perceptron_data.h"
#include "perceptron.h"

// (default) constructor
Perceptron::Perceptron(std::vector<std::vector<double> > weights, std::vector<double> bias, std::string activation) {
    SetWeights(weights);
    SetBias(bias);
    // initialize perceptron data
    // TODO: test if it rejects 'empty vectors'
    // TODO #A: replace with make_shared!
    this->_data = std::unique_ptr<PerceptronData> (new PerceptronData(weights.size(), weights[0].size(), activation));
    this->_activationFct = std::unique_ptr<ActivationFct> (new ActivationFct(activation));
}

// constructor using random initialization
// TODO #A: how to improve, i.e. directly using the perceptron without copying?
Perceptron::Perceptron(int rows, int cols, std::string activation) {
    this->_data = std::unique_ptr<PerceptronData> (new PerceptronData(rows, cols, activation));
    this->_activationFct = std::unique_ptr<ActivationFct> (new ActivationFct(activation));
    
    // TODO #A: actually need copy constructor?
    // randomly initialize perceptron from _data
    Perceptron per = _data->Initialize();
    
    SetWeights(per.Weights());
    SetBias(per.Bias());
}

// copy constructor (deep copy)
Perceptron::Perceptron(const Perceptron &perceptron) {
    _weights = perceptron._weights;
    _bias = perceptron._bias;

    *_data = *perceptron._data;
    *_activationFct = *perceptron._activationFct;
}

// copy assignment
Perceptron &Perceptron::operator=(const Perceptron &perceptron) {
    if (this == &perceptron) { // against self-assignment
        return *this;
    }
    _weights = perceptron._weights;
    _bias = perceptron._bias;

    _data.reset(new PerceptronData(*perceptron._data));
    _activationFct.reset(new ActivationFct(*perceptron._activationFct));
}


// simplify getters & setters for PerceptronData
int Perceptron::Rows() { return _data->Rows(); }
int Perceptron::Cols() { return _data->Cols(); }
std::string Perceptron::Activation() { return _data->Activation(); }

// evaluation method
std::vector<double> Perceptron::Evaluate(std::vector<double> inputVector) {
    std::vector<std::vector<double> > weights = this->Weights();
    std::vector<double> outputVector(weights.size(), 0);
    std::vector<double> bias = this->Bias();

    for (int i=0; i < this->Rows(); i++) {
        // matrix multiplication weights*inputVector
        for (int j=0; j < this->Cols(); j++) {
            outputVector[i] += weights[i][j]*inputVector[j];
        }
        // add bias 
        outputVector[i] += bias[i];
    }
    // apply activation function
   return _activationFct->Evaluate(outputVector);
}

// summary
std::string Perceptron::Summary() {
    return _data->Summary();
}