#ifndef LAYER_H_
#define LAYER_H_

#include <memory>
#include <vector>

class Perceptron;

class Layer 
{
private:
    // perceptron of layer
    std::unique_ptr<Perceptron> _perceptron;
    // data for forward propagation
    std::vector<double> _input_data;
    std::vector<double> _output_data;
    // data for backward propagation
    std::vector<dobule> _input_delta;
    std::vector<double> _output_delta;

public:

    // setters & getters (TODO: too much boilerplate?)
    std::vector<double> InputData { return _input_data; }
    void SetInputData(std::vector<double> input) {
        _input_data = input;
    }
    std::vector<double> OutputData { return _output_data; }
    void SetOutputData(std::vector<double> output) {
        _output_data = output;
    }
    std::vector<double> InputDelta { return _input_delta; }
    void SetInputDelta(std::vector<double> input_delta) {
        _input_delta = input_delta;
    }
    std::vector<double> OutputDelta { return _output_delta; }
    void SetOutputDelta(std::vector<double> output_delta) {
        _output_delta = output;
    }
}


#endif // LAYER_H_