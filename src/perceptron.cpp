#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "activation.h"
#include "perceptron_data.h"
//#include "perceptron_data.cpp" // to include perceptron_data namespace; correct way?
#include "perceptron.h"

// (default) constructor
Perceptron::Perceptron(std::vector<std::vector<double> > weights, std::vector<double> bias, std::string activation) {
    SetWeights(weights);
    SetBias(bias);
    // initialize perceptron data
    // TODO: test if it rejects 'empty vectors'
    // TODO #A: replace with make_unique!
    this->_data.reset(new PerceptronData(weights.size(), weights[0].size(), activation));
    this->_activationFct.reset(new ActivationFct(activation));
}


// constructor using random initialization
// TODO: use make_unique?
Perceptron::Perceptron(int rows, int cols, std::string activation) {
    this->_data.reset(new PerceptronData(rows, cols, activation));
    this->_activationFct.reset(new ActivationFct(activation)); 

    SetWeights(PerceptronData::WeightInitialization(rows, cols, activation));
    // TODO: suboptimal to initialize bias as zeros?
    SetBias(std::vector<double>(rows, 0));
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

    return *this;
}


// simplify getters & setters for PerceptronData
int Perceptron::Rows() { return _data->Rows(); }
int Perceptron::Cols() { return _data->Cols(); }
std::string Perceptron::Activation() { return _data->Activation(); }

// evaluation method
// TODO #A: ADD check if input works
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
std::string Perceptron::PrintDoubleVector(const std::vector<double>& double_vector) {
    std::string vector_string = "";
    for (double d: double_vector) {
        vector_string += std::to_string(d) + ",\t";
    }
    return vector_string;
}
std::string Perceptron::Summary() {
    std::string summary = _data->Summary() + "\n";
    std::string weights_string = "Weights:\n";
    std::string bias_string = "Bias:\n" + Perceptron::PrintDoubleVector(this->Bias());

    for (int i=0; i < this->Rows(); i++) {
        weights_string += Perceptron::PrintDoubleVector(this->Weights()[i]) + "\n";
    }

    return summary + weights_string + bias_string;
}