#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "layer.h"
#include "perceptron.h"
#include "sequential_nn.h"

bool ComposabilityCheck(std::vector<std::shared_ptr<Layer> > pointers_layers) {
    bool composable = true;
    int N = pointers_layers.size();

    for (int i=0; i<N-1; i++) {
        if (pointers_layers[i]->Perceptron()->Cols() != pointers_layers[i+1]->Perceptron()->Rows()) {
            composable = false;
            break;
        }
    } 
    return composable;
}

// TODO #A: efficient enough (i.e. withouth copying)?
std::vector<std::shared_ptr<Layer> > ConnectLayers(std::vector<std::shared_ptr<Layer> >& pointers_layers) {
    int N = pointers_layers.size();
    if (not ComposabilityCheck(pointers_layers)) {
            throw std::invalid_argument("Not composable!");
    } 
    else {
        for (int i=0; i<N-1; i++) {
            pointers_layers[i]->SetNext(pointers_layers[i+1]);
            pointers_layers[i+1]->SetPrevious(pointers_layers[i]);
        }
    }
    return pointers_layers;
}

SequentialNN::SequentialNN(std::vector<Layer> layers) {
    for (int i=0; i<layers.size(); i++) {
        // TODO #A: need to improve: do we copy too much here?
        _pointers_layers.emplace_back(std::make_shared<Layer>(layers[i]));
    }

    _pointers_layers = ConnectLayers(_pointers_layers);
}

std::string SequentialNN::Summary() {
    std::string summary = "Summary of sequential neural network: ";
    int N = Layers().size();

    for (int i=0; i < N; i++) {
        summary += "\nLayer " + std::to_string(i) + ":\n" + _pointers_layers[i]->Summary();
    }

    return summary;
}

void SequentialNN::Forward() {
    int N = Layers().size();
    for (int i=0; i < N; i++) {
        _pointers_layers[i]->Forward();
        //std::cout << "input (in Forward): " << Perceptron::PrintDoubleVector(_pointers_layers[i]->InputData()) << std::endl;
        //std::cout << "output (in Forward): " << Perceptron::PrintDoubleVector(_pointers_layers[i]->OutputData()) << std::endl;
    }
}

std::vector<double> SequentialNN::Evaluate(std::vector<double> input) {
    _pointers_layers[0]->SetInputData(input);
    //std::cout << "input: " << Perceptron::PrintDoubleVector((*_layers_ptr)[0].InputData()) << std::endl;
    this->Forward();
    return (_pointers_layers).back()->OutputData();
}