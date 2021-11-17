#ifndef LAYER_H_
#define LAYER_H_

#include <memory>
#include <string>
#include <vector>

class Transformation;
class LinearTransformation;

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


class LayerBase
/*
    Base of a layer in a neural network. 
    
    Functionality: 
        - keeping caches for backward and forward propagation
        - transformation of the layer
*/
{
    private:
        // cache for forward and backward pass/propagation
        std::unique_ptr<LayerCache> _forward_cache;
        std::unique_ptr<LayerCache> _backward_cache;

        // transformation
        std::unique_ptr<Transformation> _transformation;

    public:
        // default constructor
        LayerBase(Transformer);

        // setters & getters
        // for caches (note that we do not need setters)
        // they are contained in the corresponding class
        std::unique_ptr<LayerCache>& ForwardCache() { return _forward_cache; }
        std::unique_ptr<LayerCache>& BackwardCache() { return _backward_cache; }
        // NOTE: we keep the transformation 'hidden' on purpose

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

        // summary
        std::string Summary();     
};


class Layer {
    private: 
        std::shared_ptr<Layer> _next;
        std::shared_ptr<Layer> _previous;
        std::unique_ptr<LayerBase> _base;

    public:
        // default constructor
        Layer(LayerBase);

        // forward pass 
        void Forward();
        // backward pass 
        void Backward();

        // setters/getters
        // TODO: makes sense for an abstract class?!?
        // for next/previous layer
        // TODO: actually want reference?
        std::shared_ptr<Layer> Next() { return _next; }
        void SetNext(std::shared_ptr<Layer> next); 
        std::shared_ptr<Layer> Previous() { return _previous; }
        void SetPrevious(std::shared_ptr<Layer> pointer_previous);
        // for LayerBase
        std::unique_ptr<LayerBase>& Base() { return _base; }

        // summary
        std::string Summary();
};

class LinearLayer : public Layer {
    public:
        // default constructor
        LinearLayer(LinearTransformation):
            Layer(LinearTransformation) 
            {
            }
        // simplified constructor
        LinearLayer(int, int);

        // update weights (gradient descent)
        void UpdateWeights();
};

class 

// flatten layer --> extra layer class
        //static std::vector<double> Flatten(const std::vector<std::vector<double> >& matrix);



#endif // LAYER_H_