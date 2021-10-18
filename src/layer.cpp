#include <vector>
#include "layer.h"
#include "perceptron_data.h"
#include "perceptron.h"


// setters & getters
std::vector<double> LayerBase::InputData() { return _input_data; }
void LayerBase::SetInputData(std::vector<double> input) {
        _input_data = input;
}

std::vector<double> LayerBase::OutputData() { return _output_data; }
void LayerBase::SetOutputData(std::vector<double> output) {
        _output_data = output;
}

std::vector<double> LayerBase::InputDelta() { return _input_delta; }
void LayerBase::SetInputDelta(std::vector<double> input_delta) {
        _input_delta = input_delta;
}

std::vector<double> LayerBase::OutputDelta() { return _output_delta; }
void LayerBase::SetOutputDelta(std::vector<double> output_delta) {
        _output_delta = output_delta;
}


// constructor for LayerBase
LayerBase::LayerBase(int rows, int cols, std::string activation) {
        this->_perceptron = std::unique_ptr<Perceptron> (new Perceptron(rows, cols, activation));
        // TODO #A: how to use move semantics?
        std::vector<double> input(this->_perceptron->Cols(), 0);
        std::vector<double> output(this->_perceptron->Rows(), 0);
        SetInputData(input);
        SetInputDelta(input);
        SetOutputData(output);
        SetOutputDelta(output);
}

