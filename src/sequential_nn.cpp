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

    for (int i=0; i < (*_layers_ptr).size(); i++) {
        summary += "\nLayer " + std::to_string(i) + ": " + (*_layers_ptr)[i].Summary();
    }

    return summary;
}