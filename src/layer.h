#ifndef LAYER_H_
#define LAYER_H_

#include <memory>
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
    std::unique_ptr<Perceptron> _perceptron;
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
    std::vector<double> InputData(); 
    void SetInputData(std::vector<double>); 

    std::vector<double> OutputData(); 
    void SetOutputData(std::vector<double>); 

    std::vector<double> InputDelta(); 
    void SetInputDelta(std::vector<double>);

    std::vector<double> OutputDelta(); 
    void SetOutputDelta(std::vector<double>);

    // perceptron data
    int Rows() { return _perceptron->Rows(); }
    int Cols() { return _perceptron->Cols(); }
    std::vector<std::vector<double> > Weights() { return _perceptron->Weights(); }
    std::vector<double> Bias() { return _perceptron->Bias(); }

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
        LayerBase _layer_base;

        // NOTE: even though we only implement sequential NNs, we need shared
        // pointers because a hidden layer is pointed at twice (as _next and
        // _previous).
        // next layer
        std::shared_ptr<Layer> _next(nullptr); 
        // previous layer
        std::shared_ptr<Layer> _previous(nullptr);

    public:
        // constructor
        // TODO !!!

        // setters & getters
        std::shared_ptr<LayerBase> Next() { return _next; }
        void SetNext(LayerBase next) { _next = next; }
        LayerBase Previous() { return _previous; }
        void SetNext(LayerBase previous) { _previous = previous; }
};




#endif // LAYER_H_