#include "layer_cache.h"
#include <memory>
#include <vector>

/**************
 * LAYER CACHE *
 **************/

// TODO: would it be better to give other options to set the corresponding vectors?

void LayerCache::SetForwardInput(std::shared_ptr<Eigen::VectorXd> input_ptr) {
    _forward_input = std::move(input_ptr);
}

void LayerCache::SetForwardOutput(std::shared_ptr<Eigen::VectorXd> output_ptr) {
    _forward_output = std::move(output_ptr);
}

std::shared_ptr<Eigen::VectorXd> LayerCache::GetForwardOutput() {
    return _forward_output;
}

std::shared_ptr<Eigen::VectorXd> LayerCache::GetForwardInput() {
    return _forward_input;
}

void LayerCache::SetBackwardInput(std::shared_ptr<Eigen::VectorXd> backward_input_ptr) {
    _backward_input = std::move(backward_input_ptr);
}

std::shared_ptr<Eigen::VectorXd> LayerCache::GetBackwardOutput() { 
    return _backward_output;
}

void LayerCache::ConnectForward(int size_of_vector, LayerCache& next_layer_cache) {
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(size_of_vector);

    this->SetForwardOutput(std::make_shared<Eigen::VectorXd>(zero_vector));
    next_layer_cache.SetForwardInput(this->GetForwardOutput());
}