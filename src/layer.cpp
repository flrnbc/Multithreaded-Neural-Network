#include <vector>
#include "layer.h"
#include "perceptron_data.h"
#include "perceptron.h"


// setters & getters
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
        this->_perceptron = std::unique_ptr<Perceptron> (new Perceptron(rows, cols, activation));
        // TODO #A: should we use move semantics here?
        std::vector<double> input(this->_perceptron->Cols(), 0);
        std::vector<double> output(this->_perceptron->Rows(), 0);
        SetInputData(input);
        SetInputDelta(input);
        SetOutputData(output);
        SetOutputDelta(output);
}


void Layer::SetNext(LayerBase next) {
        *_next = next; 
}

void Layer::SetPrevious(LayerBase previous) { 
        *_previous = previous; 
}



// forward pass
//void Layer::UpdateInput() 

