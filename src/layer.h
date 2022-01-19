#ifndef LAYER_H_
#define LAYER_H_

#include <Eigen/Dense>
#include "layer_cache.h"
#include "transformation.h"
#include <string>

/*********
 * LAYER *
 *********/

/** 
    Class which represents a layer in a neural network and allows the forward pass
    of data (done with a Transformation) and the backward pass (essentially with the
    derivative of the Transformation).

    The two member variables of the abstract base class are
        * _transformation_: shared pointer to the Transformation of the corresponding layer,
        * _layer_cache: shared pointer to a LayerCache to store the data of the forward and backward pass.

    The most important member functions are
        * ~Input~: sets the vector in forward input of the LayerCache
        * ~Forward~: applies the transformation to the input and stores it in forward output
        * ~Output~: returns the vector in forward output
        
        * ~UpdateDerivative~: updates the derivative of the transformation at the vector in forward output
        * ~BackwardInput~, ~Backward~ and ~BackwardOutput: work analogously as ~Input~, ~Forward~ and ~Output~
          only with the derivative of the transformation
    
    NOTE: some of the member variables, e.g. ~Cols~, ~Rows~, ~UpdateWeights~, just simplify the access
    to the corresponding member functions of Transformation.

    NOTE: Both the forward pass (using ~Forward~) and backward pass (using ~Backward~)
    transform the data and update the corresponding LayerCaches. In particular, applying
    ~Forward~ and ~Backward~ does not return a vector. Instead we have to use ~Output~ and
    ~BackwardOutput~. This is ok because the user usually does not work with Layers directly.


*/

// forward declarations
// TODO: really needed here?
class LayerCache;
class Transformation;
class LinearTransformation;
class ActivationTransformation;

class Layer {
    protected:
        // only protected to make access to data members easier in concrete derived classes
        std::shared_ptr<LayerCache> _layer_cache;
        
        // transformation for forward pass
        std::shared_ptr<Transformation> _transformation;
        // TODO: better would be to include a std::unique_ptr<Transformation> _transformation.
        // But this caused some issues (e.g. could not make _transformation.Transform() work).

        // constructor (needed for all concrete derived classes)
        Layer(std::shared_ptr<Transformation> transformation): 
            _layer_cache(std::make_shared<LayerCache>()),
            _transformation(transformation)
            {}

    public:
        // destructor 
        virtual ~Layer() {}

        // setter
        void SetLayerCache(std::unique_ptr<LayerCache>);

        // getters
        // NOTE: this is safe because we initialize _layer_cache (not as a nullptr) in 
        // concrete derived classes
        LayerCache& GetLayerCache() { return *_layer_cache; }
        std::shared_ptr<Transformation>& GetTransformation() { return _transformation; }
        // TODO: this might be confusing though because GetTransformation returns a shared_ptr...
        
        // getters to simplify access to data members of transformation
        int Cols() { return _transformation->Cols(); }
        int Rows() { return _transformation->Rows(); }
        std::string Summary() { return _transformation->Summary(); } // TODO: this might change in the future to contain more specific data of the layer

        // forward pass
        void Input(Eigen::VectorXd); // TODO: do we copy too often here?
        void Forward();
        Eigen::VectorXd Output(); // TODO: better to return const ref?

        // backward pass
        void UpdateDerivative();
        void BackwardInput(Eigen::RowVectorXd);
        void Backward();
        Eigen::RowVectorXd BackwardOutput();

        // update weights/bias (empty for ActivationLayers)
        virtual void UpdateWeights(double learning_rate) {}
        virtual void UpdateBias(double learning_rate) {}
};


/**********************
 * LINEAR LAYER CLASS *
 **********************/
 
// NOTE: possible IDEA for refactoring: use a factory pattern for creating various layers?

class LinearLayer: public Layer {
    public:
        // constructor
        LinearLayer(int rows, int cols): 
            Layer(std::make_shared<LinearTransformation>(rows, cols)) {}
            
        // initialize weights
        void Initialize(std::string initialization_type);

        // update weights and bias
        void UpdateWeights(double) override;
        void UpdateBias(double) override;
};


/********************
 * ACTIVATION LAYER *
 ********************/

class ActivationLayer: public Layer {
    private:
        std::string _activation;

    public:
        // constructor
        ActivationLayer(int m, std::string activation_fct): 
            Layer(std::make_shared<ActivationTransformation>(m, activation_fct)),
            _activation(activation_fct) 
            {}

        // getters
        std::string GetActivationString() { return _activation; }
};

#endif // LAYER_H_