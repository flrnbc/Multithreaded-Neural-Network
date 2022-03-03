#include <algorithm>
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "layer.h"
#include "layer_cache.h"
#include "loss_function.h"
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
    std::string layer_type = layer->GetTransformation()->Type();
    std::string next_layer_type = next_layer->GetTransformation()->Type();
    
    if (layer_type != "LinearTransformation") { // TODO: this is not ideal if we e.g. add further layers
        return "";
    }

    std::vector<std::string> approximate_linear = {"identity", "sigmoid", "tanh", "softmax"};
    std::vector<std::string> other_activation = {"relu", "prelu"};

    // search for different activations
    // TODO: couldn't this be optimized? (e.g. consider case where next_layer_transformation_type == "LinearTransformation")
    auto find_approximate = std::find(std::begin(approximate_linear), std::end(approximate_linear), next_layer_type);
    auto find_other = std::find(std::begin(other_activation), std::end(other_activation), next_layer_type);

    if (find_approximate != std::end(approximate_linear)) {
            return "Xavier";
        } 
    
    else if (find_other != std::end(other_activation)) {
                return "He";
            }
    
    return "";
}


/********************************************
 * SEQUENTIAL NEURAL NETWORK IMPLEMENTATION *
 ********************************************/

// constructor
SequentialNN::SequentialNN(std::vector<std::shared_ptr<Layer> > layers) {
    if (ComposabilityCheck(layers) == false) {
        throw std::invalid_argument("Layers are not composable.");
    } else {
        int N = layers.size();

        for (int i = 0; i < N; i++) {
            _layers.emplace_back(std::move(layers[i])); // TODO: move necessary?
        }

        // connect layers
        for (int i = 0; i < N-1; i++) {
            // only one sample for initialization/connecting the layers
            int batchSize = 1;
            _layers[i]->GetLayerCache().Connect(_layers[i]->Rows(), batchSize, _layers[i+1]->Cols(), _layers[i+1]->GetLayerCache());
        }
    }
    // finally initialize the weights
    Initialize();
}

// getters
int SequentialNN::InputSize() {
    if (_layers[0] == nullptr || _layers.empty()) {
        throw std::invalid_argument("Layers are empty.");
    }
    return _layers[0]->Cols();
}
int SequentialNN::OutputSize() {
    if (_layers[0] == nullptr || _layers.empty()) {
        throw std::invalid_argument("Layers are empty.");
    }
    return _layers.back()->Rows();
}

// methods
void SequentialNN::Initialize() {
    int N = _layers.size();

    for (int i=0; i < N-1; i++) {
        _layers[i]->GetTransformation()->Initialize(SequentialNN::GetInitializationType(_layers[i], _layers[i+1]));
    }
    // TODO: here might be some room to optimize (e.g. jump over the ones which have "" as initialization type)
}

std::string SequentialNN::Summary() {
    std::string summary = "Summary of sequential neural network: \n";
    int N = _layers.size();

    for (int i=0; i < N; i++) {
        summary += "\nLayer " + std::to_string(i) + ":\n" + _layers[i]->Summary() + "\n";
    }

    return summary;
}


/****************
 * FORWARD PASS *
 ****************/
void SequentialNN::Input(Eigen::MatrixXd input) {
    _layers[0]->Input(input);
    // now adjust the batch size if necessary
    int batchSize = input.cols();
    for (int i=0; i < Length(); i++) {
        _layers[i]->GetLayerCache().SetBatchSize(batchSize); 
    }
}

void SequentialNN::Forward() {
    for (int i=0; i < Length(); i++) {
        _layers[i]->Forward();
    }
}
Eigen::MatrixXd& SequentialNN::Output() {
    if ((_layers.back())->GetLayerCache().GetForwardOutput() == nullptr) {
        throw std::invalid_argument("Forward output is null.");
    }
    return *((_layers.back())->GetLayerCache().GetForwardOutput());
}
Eigen::MatrixXd SequentialNN::operator()(Eigen::MatrixXd input) {
    Input(input);
    Forward();
    return Output();
}


/*****************
 * BACKWARD PASS *
 *****************/
void SequentialNN::BackwardInput(Eigen::MatrixXd backward_input) {
    _layers.back()->BackwardInput(backward_input);
    // adjust the batch size if necessary 
    int batchSize = backward_input.rows();
    for (int i=0; i < Length(); i++) {
        _layers[i]->GetLayerCache().SetBatchSize(batchSize); 
    }
}
void SequentialNN::Backward() {
    for (int i = Length()-1; i >= 0; i--) {
        _layers[i]->Backward();
    }
}
Eigen::MatrixXd SequentialNN::BackwardOutput() {
     if (_layers[0]->GetLayerCache().GetBackwardOutput() == nullptr) {
        throw std::invalid_argument("Backward output is null.");
    }
    return *(_layers[0]->GetLayerCache().GetBackwardOutput());
}

/*************************
 * UPDATE WEIGHTS & BIAS *
 *************************/
void SequentialNN::UpdateWeightsBias(double learning_rate=1.0) {
    for (int i = 0; i < Length(); i++) {
        _layers[i]->UpdateWeightsBias(learning_rate);
    }
}

