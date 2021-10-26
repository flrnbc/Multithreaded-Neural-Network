#include <iostream>
#include <stdexcept>
#include <vector>

#include "layer.h"
#include "perceptron.h"
#include "sequential_nn.h"

bool ComposabilityCheck(std::vector<Layer> layers) {
    bool composable = true;
    for (int i=0; i<layers.size()-1; i++) {
        if (layers[i].Perceptron()->Cols() != layers[i+1].Perceptron()->Rows()) {
            composable = false;
            break;
        }
    } 
    return composable;
}

std::vector<Layer> ConnectLayers(std::vector<Layer> layers) {
    if (not ComposabilityCheck(layers)) {
            throw std::invalid_argument("Not composable!");
    } 
    else {
        for (int i=0; i<layers.size()-1; i++) {
            layers[i].SetNext(layers[i+1]);
            layers[i+1].SetPrevious(layers[i]);
        }
    }
    return layers;
}

SequentialNN::SequentialNN(std::vector<Layer> layers) {
    std::vector<Layer> connectedLayers = ConnectLayers(layers);
    // make_unique did not work...
    this->_layers_ptr.reset(new std::vector<Layer>(connectedLayers));
}

std::string SequentialNN::Summary() {
    std::string summary = "Summary of sequential neural network: ";

    for (int i=0; i < _layers_ptr->size(); i++) {
        summary += "\nLayer " + std::to_string(i) + ":\n" + (*_layers_ptr)[i].Summary();
    }

    return summary;
}

void SequentialNN::Forward() {
    for (int i=0; i < _layers_ptr->size(); i++) {
        (*_layers_ptr)[i].Forward();
        std::cout << "input (in Forward): " << Perceptron::PrintDoubleVector((*_layers_ptr)[i].InputData()) << std::endl;
        std::cout << "output (in Forward): " << Perceptron::PrintDoubleVector((*_layers_ptr)[i].OutputData()) << std::endl;
    }
}

std::vector<double> SequentialNN::Evaluate(std::vector<double> input) {
    (*_layers_ptr)[0].SetInputData(input);
    //std::cout << "input: " << Perceptron::PrintDoubleVector((*_layers_ptr)[0].InputData()) << std::endl;
    this->Forward();
    return (*_layers_ptr).back().OutputData();
}