#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "layer.h"
#include "layer_cache.h"
#include "transformation.h"
#include "sequential_nn.h"

/********************
 * HELPER FUNCTIONS *
 ********************/

bool SequentialNN::ComposabilityCheck(const std::vector<std::shared_ptr<Layer> >& layers) {
    bool composable = true;
    int N = layers.size();

    for (int i=0; i<N-1; i++) {
        if (layers[i]->Rows() != layers[i+1]->Cols()) {
            composable = false;
            break;
        }
    } 
    return composable;
}

std::string SequentialNN::GetInitializationType(const std::shared_ptr<Layer>& layer, const std::shared_ptr<Layer>& next_layer) {
    std::string layer_transformation_type = layer->GetTransformation()->Type();
    std::string next_layer_transformation_type = next_layer->GetTransformation()->Type();
    
    if (layer_transformation_type != "LinearTransformation") { // TODO: this is not ideal if we e.g. add further layers
        return "";
    }

    std::vector<std::string> approximate_linear_activation = {"identity", "sigmoid", "tanh"};
    std::vector<std::string> other_activation = {"relu", "prelu"};

    // TODO: couldn't this be optimized? (e.g. consider case where next_layer_transformation_type == "LinearTransformation")
    if (std::find(std::begin(approximate_linear_activation), 
        std::end(approximate_linear_activation), 
        next_layer_transformation_type) != std::end(approximate_linear_activation)) {
            return "Xavier";
        } 
    
    else if (std::find(std::begin(other_activation), 
            std::end(other_activation), 
            next_layer_transformation_type) != std::end(other_activation)) {
                return "He";
            }
    return "";
}


/********************************************
 * SEQUENTIAL NEURAL NETWORK IMPLEMENTATION *
 ********************************************/

SequentialNN::SequentialNN(std::vector<std::shared_ptr<Layer> > layers) {
    if (ComposabilityCheck(layers) == false) {
        throw std::invalid_argument("Layers are not composable.");
    } else {
        int N = layers.size();

        for (int i = 0; i < N; i++) {
            _layers.emplace_back(std::move(layers[i])); // TODO: move necessary?
        }

        for (int i = 0; i < N-1; i++) {
            _layers[i]->GetLayerCache().ConnectForward(_layers[i]->Rows(), _layers[i+1]->GetLayerCache());
        }
    }
}

void SequentialNN::Initialize() {
    int N = _layers.size();

    for (int i = 0; i < N-1; i++) {
        _layers[i]->GetTransformation()->Initialize(SequentialNN::GetInitializationType(_layers[i], _layers[i+1]));
    }
    // TODO: here might be some room to optimize (e.g. jump over the ones which have "" as initialization type)
}

std::string SequentialNN::Summary() {
    std::string summary = "Summary of sequential neural network: ";
    int N = _layers.size();

    for (int i=0; i < N; i++) {
        summary += "\nLayer " + std::to_string(i) + ":\n" + _layers[i]->Summary();
    }

    return summary;
}

void SequentialNN::Forward() {
    for (int i = 0; i < Length(); i++) {
        _layers[i]->Forward();
        //std::cout << "input (in Forward): " << Perceptron::PrintDoubleVector(_pointers_layers[i]->InputData()) << std::endl;
        //std::cout << "output (in Forward): " << Perceptron::PrintDoubleVector(_pointers_layers[i]->OutputData()) << std::endl;
    }
}

void SequentialNN::Input(std::vector<double> input) {
    _layers[0]->Input(input);
}

std::vector<double> SequentialNN::Output() {
    if ((_layers.back())->GetLayerCache().GetForwardOutput() == nullptr) {
        throw std::invalid_argument("Backward output is null.");
    }
    
    return *((_layers.back())->GetLayerCache().GetForwardOutput());
}