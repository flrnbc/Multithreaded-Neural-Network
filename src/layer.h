#ifndef LAYER_H_
#define LAYER_H_

#include "layer_cache.h"
#include "transformation.h"
#include <string>
#include <vector>

class LayerCache;
class Transformation;
class LinearTransformation;
class ActivationTransformation;

class Layer {
    protected:
        // only protected to make access to layer cache easier
        std::shared_ptr<LayerCache> _layer_cache;
        // transformation of layer
        std::shared_ptr<Transformation> _transformation;
        // TODO: better would be to include a std::unique_ptr<Transformation> _transformation.
        // But this caused some issues (e.g. could not make _transformation.Transform() work).

        Layer(std::shared_ptr<Transformation> transformation): 
            _layer_cache(std::make_shared<LayerCache>()),
            _transformation(transformation)
            {}

    public:
        // destructor 
        virtual ~Layer() {}

        // NOTE: we only want to move Layers into std::vectors so that we 
        // only implement move semantics. Following the rule of five, we delete
        // the copy operators.
        // TODO: might change in the future, if we want to be able to copy layers etc. ...

        // copy constructor
        // Layer(const Layer &source) {
        //     _layer_cache = std::make_unique<LayerCache>(*(source._layer_cache)); 
        // }

        // // copy assignment operator
        // Layer& operator=(const Layer &source) {
        //     if (this == &source) {
        //         return *this;
        //     }
        //     _layer_cache = std::make_unique<LayerCache>(*(source._layer_cache));

        //     return *this;
        // }
       
        // // move constructor
        // Layer(Layer &&source) {
        //     _layer_cache = std::move(source._layer_cache); // just moves the unique_ptr
        //     // TODO: hence no need to set _layer_cache to null?
        // }

        // // move assignment operator
        // Layer& operator=(Layer &&source) {
        //     if (this == &source) {
        //         return *this;
        //     }
        //     _layer_cache = std::move(source._layer_cache);

        //     return *this;
        // }

        // setter
        void SetLayerCache(std::unique_ptr<LayerCache>);

        // getters
        // TODO: better to return constant ref to the underlying raw pointer?
        std::shared_ptr<LayerCache>& GetLayerCache() { return _layer_cache; }
        std::shared_ptr<Transformation> GetTransformation() { return _transformation; }
        
        // getters to simplify access to data members of transformation
        int Cols() { return _transformation->Cols(); }
        int Rows() { return _transformation->Rows(); }
        std::string Summary() { return _transformation->Summary(); } // TODO: this might change in the future to contain more specific data

        // forward pass
        void Input(std::vector<double>); // TODO: do we copy too often here?
        virtual void Forward() = 0;
        std::vector<double> Output(); // TODO: better to return const ref?

        // backward pass
        // virtual void Backward() = 0;
};

/**********************
 * LINEAR LAYER CLASS *
 **********************/
 
// TODO: need to refactor, e.g. with an interface Layer or a factory method?
// issues with interface Layer: Transformation class and downcasting

class LinearLayer: public Layer {
    //private:
    //    std::shared_ptr<LinearTransformation> _transformation;

    public:
        // constructor
        LinearLayer(int rows, int cols): 
            Layer(std::make_shared<LinearTransformation>(rows, cols)) {}
            
        // destructor
        // TODO: always needed?
        ~LinearLayer() {};

        // initialize
        void Initialize(std::string initialization_type);

        // forward pass
        void Forward();
        // backward pass AND updating weights
        //void Backward() {}
};


/********************
 * ACTIVATION LAYER *
 ********************/

class ActivationLayer: public Layer {
    private:
    //    std::shared_ptr<ActivationTransformation> _transformation;
        std::string _activation;

    public:
        // default constructor
        ActivationLayer(int m, std::string activation_fct): 
            Layer(std::make_shared<ActivationTransformation>(m, activation_fct)),
            _activation(activation_fct) 
            {}
           
        // destructor
        // TODO: always needed?
        ~ActivationLayer() {};

        // setters/getters
        std::string GetActivationString() { return _activation; }

        // forward pass
        void Forward();
        // backward pass AND updating weights
        //void Backward() {}
};



// flatten layer --> extra layer class
        //static std::vector<double> Flatten(const std::vector<std::vector<double> >& matrix);



#endif // LAYER_H_