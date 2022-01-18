#include "layer_cache.h"
#include <memory>
#include <vector>

/**************
 * LAYER CACHE *
 **************/

// TODO: would it be better to give other options to set the corresponding vectors?

// setters/getters
// foward 
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

// backward
void LayerCache::SetBackwardInput(std::shared_ptr<Eigen::RowVectorXd> backward_input_ptr) {
    _backward_input = std::move(backward_input_ptr);
}
void LayerCache::SetBackwardOutput(std::shared_ptr<Eigen::RowVectorXd> backward_output_ptr) {
    _backward_output = std::move(backward_output_ptr);
}

std::shared_ptr<Eigen::RowVectorXd> LayerCache::GetBackwardInput() { 
    return _backward_input;
}
std::shared_ptr<Eigen::RowVectorXd> LayerCache::GetBackwardOutput() { 
    return _backward_output;
}


// connecting layer caches ('from left to right')
// forward
void LayerCache::ConnectForward(int size_forward_output, LayerCache& next_layer_cache) {
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(size_forward_output);

    // TODO: might need a check if SetForwardOutput is not null, e.g. if we add 
    // the option to add/delete layers
    this->SetForwardOutput(std::make_shared<Eigen::VectorXd>(zero_vector));
    next_layer_cache.SetForwardInput(this->GetForwardOutput());
}

// backward
// NOTE: since we propagate backwards, the backward input of *this points to the same
// vector as the backward output of next_layer_cache.
void LayerCache::ConnectBackward(int size_backward_input, LayerCache& next_layer_cache) {
    Eigen::RowVectorXd zero_vector = Eigen::RowVectorXd::Zero(size_backward_input);

    this->SetBackwardInput(std::make_shared<Eigen::RowVectorXd>(zero_vector));
    next_layer_cache.SetBackwardOutput(this->GetBackwardInput());
}

// connect both
void LayerCache::Connect(int size_forward_output, int size_backward_input, LayerCache& next_layer_cache) {
    ConnectForward(size_forward_output, next_layer_cache);
    ConnectBackward(size_backward_input, next_layer_cache);
}
