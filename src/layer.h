#ifndef LAYER_H_
#define LAYER_H_

#include <memory>
#include <string>
#include <vector>

class Perceptron;

class LayerBase
/*
    Class which enhances a perceptron with input/output data 
    as well as input/output deltas for forward and backward
    propagation respectively.
*/
{
private:
    // perceptron of layer
    // NOTE: a perceptron is obviously (uniquely) tied to an object of LayerBase.
    // however, when we built neural networks from Layer/LayerBase, the perceptron
    // is shared by connected layers. hence we use a shared_ptr.
    std::shared_ptr<Perceptron> _perceptron;
    // data for forward propagation
    std::vector<double> _input_data;
    std::vector<double> _output_data;
    // data for backward propagation
    std::vector<double> _input_delta;
    std::vector<double> _output_delta;

public:
    // TODO: default constructor

    // constructor 
    LayerBase(int, int, std::string);

    // setters & getters (TODO: too much boilerplate?)
    // input and output data
    std::vector<double> InputData() { return _input_data; } 
    void SetInputData(std::vector<double>); 
    
    std::vector<double> OutputData() { return _output_data; }
    void SetOutputData(std::vector<double>); 

    std::vector<double> InputDelta() { return _input_delta; }
    void SetInputDelta(std::vector<double>);

    std::vector<double> OutputDelta() { return _output_delta; }
    void SetOutputDelta(std::vector<double>);

    // perceptron 
    std::shared_ptr<Perceptron> Perceptron() { return _perceptron; }
    // int Rows() { return _perceptron->Rows(); }
    // int Cols() { return _perceptron->Cols(); }
    // std::vector<std::vector<double> > Weights() { return _perceptron->Weights(); }
    // std::vector<double> Bias() { return _perceptron->Bias(); }

    // summary
    std::string Summary();
};


class Layer : public LayerBase
/*
    Class of a layer in a neural network based on LayerBase.
    It enhances LayerBase by the neighbors (_next and _previous)
    in a neural network. 

    TODO #A: does nullptr help to implement input/output layers?
*/
{
    private:
        // NOTE: even though we only implement sequential NNs, we need shared
        // pointers because a hidden layer is pointed at twice (as _next and
        // _previous). Also note that smart pointers are initialized to nullptr
        // by default.

        // next layer
        std::shared_ptr<LayerBase> _next; 
        // previous layer
        std::shared_ptr<LayerBase> _previous;

    public:
        // constructor
        Layer(int rows, int cols, std::string activation);

        // setters & getters
        std::shared_ptr<LayerBase> Next() { return _next; }
        void SetNext(LayerBase next); 
        std::shared_ptr<LayerBase> Previous() { return _previous; }
        void SetPrevious(LayerBase previous);

        // forward pass
        void UpdateInput(); 
        void Forward(); // in particular updates output

        // backward pass
        // TODO #A: ADD!!!
};




#endif // LAYER_H_