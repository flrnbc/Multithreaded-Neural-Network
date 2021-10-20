#include <iostream>
#include <vector>

#include "layer.h"
#include "perceptron.h"
#include "sequential_nn.h"

SequentialNN::SequentialNN(std::vector<Layer> layers) {
    // check if perceptrons are composable
    for (int i=0; i++; i<layers.size()-1) {
        if (layers[i].Perceptron()->Rows() == layers[i+1].Perceptron()->Cols()) {
            layers[i].SetNext(layers[i+1]);
            layers[i+1].SetPrevious(layers[i]);
        } else {
            std::cout << "Perceptrons not composable." << std::endl;
        }
    }
}

std::string SequentialNN::Summary() {
    std::string summary = "Summary of sequential neural network: " + '\n';

    for (int i=0; i++; i < (*_layers_ptr).size()) {
        summary += "Layer " + std::to_string(i) + ": " + (*_layers_ptr)[i].Summary();
    }

    return summary;
}