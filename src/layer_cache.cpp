#include "layer_cache.h"
#include <memory>
#include <vector>

/**************
 * LAYER CACHE *
 **************/

// TODO: would it be better to give other options to set the corresponding vectors?

void LayerCache::SetForwardInput(std::shared_ptr<std::vector<double> > input_ptr) {
    _forward_input = std::move(input_ptr);
}

void LayerCache::SetForwardOutput(std::shared_ptr<std::vector<double> > output_ptr) {
    _forward_output = std::move(output_ptr);
}

std::shared_ptr<std::vector<double> > LayerCache::GetForwardOutput() {
    return _forward_output;
}

std::shared_ptr<std::vector<double> > LayerCache::GetForwardInput() {
    return _forward_input;
}

void LayerCache::SetBackwardInput(std::shared_ptr<std::vector<double> > backward_input_ptr) {
    _backward_input = std::move(backward_input_ptr);
}

std::shared_ptr<std::vector<double> > LayerCache::GetBackwardOutput() { 
    return _backward_output;
}

void LayerCache::ConnectForward(int size_of_vector, LayerCache& next_layer_cache) {
    std::vector<double> zero_vector(size_of_vector, 0);

    this->SetForwardOutput(std::make_shared<std::vector<double> >(zero_vector));
    next_layer_cache.SetForwardInput(this->GetForwardOutput());
}