#ifndef LAYER_H_
#define LAYER_H_

#include <memory>
#include <string>
#include <vector>

class Perceptron
class Transformation;

// TODO #A: use a template (e.g. for matrices etc.)
class Cache {
    private:
        std::vector<double> _cached_vector;
    public:
        // (default) constructor
        Cache(std::vector<double>);
        // alternative constructor (initialize with zero vector of given size)
        Cache(int input_size);
        
        // setters/getters
        std::vector<double> CachedVector() { return _cached_vector; } 
        void SetCachedVector(std::vector<double>); 

        // reset cached vector (to zero)
        void Reset();
};


class LayerCache {
    private:
        std::shared_ptr<Cache> _input_cache;
        std::shared_ptr<Cache> _output_cache;
    public:
        std::shared_ptr<Cache> Input() { return _input_cache; }
        std::shared_ptr<Cache> Output() { return _output_cache; }
        // reset
        void Reset();
};


class Layer
/*
    Class of a layer in a neural network based on LayerBase.
    It enhances LayerBase by the neighbors (_next and _previous)
    in a neural network. 
*/
{
    private:
        // transformation of layer
        // TODO: check if unique_ptr makes sense/requires copy constructor etc.
        std::unique_ptr<Transformation> _transformation;
        // cache for forward pass
        std::unique_ptr<LayerCache> _forward_cache;
        // cache for backward pass
        std::unique_ptr<LayerCache> _backward_cache;
        // next layer
        std::shared_ptr<Layer> _next; 
        // previous layer
        std::shared_ptr<Layer> _previous;

    public:
        // (default) constructor 
        Layer(Transformation);

        // setters & getters
        // for next/previous layer
        // TODO: actually want reference?
        std::shared_ptr<Layer> Next() { return _next; }
        void SetNext(std::shared_ptr<Layer> next); 
        std::shared_ptr<Layer> Previous() { return _previous; }
        void SetPrevious(std::shared_ptr<Layer> pointer_previous);
        // for caches
        std::unique_ptr<LayerCache>& ForwardCache() { return _forward_cache; }
        std::unique_ptr<LayerCache>& BackwardCache() { return _backward_cache; }
        
        // 'forward propagation'
        // update input of foward cache (i.e. take output from previous layer)
        void UpdateInput();
        // update output of forward cache (i.e. transform input and store in output)
        void UpdateOutput();

        // backward propagation
        // update input of backward cache (take output from previous layer)
        void UpdateBackwardInput();
        // update output of backward cache
        void UpdateBackwardOutput();
        
};









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
        // TODO #A: is it inefficient to use this method to evaluate a layer?
        // return by value vs. by ref
        std::shared_ptr<Perceptron> Perceptron() { return _perceptron; }
        // int Rows() { return _perceptron->Rows(); }
        // int Cols() { return _perceptron->Cols(); }
        // std::vector<std::vector<double> > Weights() { return _perceptron->Weights(); }
        // std::vector<double> Bias() { return _perceptron->Bias(); }

        // update output (i.e. evaluate)
        void UpdateOutput();
        // summary
        std::string Summary();
};

// flatten layer --> extra layer class
        //static std::vector<double> Flatten(const std::vector<std::vector<double> >& matrix);



#endif // LAYER_H_