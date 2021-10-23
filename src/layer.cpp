#include <string>
#include <vector>
#include "layer.h"
#include "perceptron_data.h"
#include "perceptron.h"

// LayerBase //
// setters & getters for LayerBase
void LayerBase::SetInputData(std::vector<double> input) {
        _input_data = input;
}
void LayerBase::SetOutputData(std::vector<double> output) {
        _output_data = output;
}
void LayerBase::SetInputDelta(std::vector<double> input_delta) {
        _input_delta = input_delta;
}
void LayerBase::SetOutputDelta(std::vector<double> output_delta) {
        _output_delta = output_delta;
}

// constructor for LayerBase
LayerBase::LayerBase(int rows, int cols, std::string activation) {
        this->_perceptron = std::shared_ptr<Perceptron> (new Perceptron(rows, cols, activation));
        // TODO #A: should we use move semantics here?
        std::vector<double> input(this->_perceptron->Cols(), 0);
        std::vector<double> output(this->_perceptron->Rows(), 0);
        SetInputData(input);
        SetInputDelta(input);
        SetOutputData(output);
        SetOutputDelta(output);
}

// summary
std::string LayerBase::Summary() {
        return _perceptron->Summary();
}


// Layer //
// setters & getters for Layer
void Layer::SetNext(LayerBase next) {
        *_next = next; 
}

void Layer::SetPrevious(LayerBase previous) { 
        *_previous = previous; 
}

// constructor for Layer
Layer::Layer(int rows, int cols, std::string activation) : 
        LayerBase(rows, cols, activation), 
        _next(nullptr),
        _previous(nullptr) {}

// forward pass
// NOTE: no need to update for input layer (need to set input though)
void Layer::UpdateInput() {
        if (Previous() != nullptr) {
                SetInputData(Previous()->OutputData());
        }
}

void Layer::Forward() {
        SetOutputData(Perceptron()->Evaluate(InputData()));
}

// backward pass
// TODO #A: ADD!!!

