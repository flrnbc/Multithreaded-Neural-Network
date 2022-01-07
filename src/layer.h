#ifndef LAYER_H_
#define LAYER_H_

#include <Eigen/Dense>
#include "layer_cache.h"
#include "transformation.h"
#include <string>

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
        // TODO: this might be confusing though because GetTransformation returns a shared_ptr...
        LayerCache& GetLayerCache() { return *_layer_cache; }
        std::shared_ptr<Transformation>& GetTransformation() { return _transformation; }
        
        // getters to simplify access to data members of transformation
        int Cols() { return _transformation->Cols(); }
        int Rows() { return _transformation->Rows(); }
        std::string Summary() { return _transformation->Summary(); } // TODO: this might change in the future to contain more specific data

        // forward pass
        void Input(Eigen::VectorXd); // TODO: do we copy too often here?
        void Forward(); // TODO: use _transformation to rewrite in a coherent way, not for each layer type
        Eigen::VectorXd Output(); // TODO: better to return const ref?

        // backward pass
        void UpdateDerivative();
        void BackwardInput(Eigen::RowVectorXd);
        void Backward();
        Eigen::RowVectorXd BackwardOutput();
};


/**********************
 * LINEAR LAYER CLASS *
 **********************/
 
// possible IDEA for refactoring: use a factory pattern for creating various layers?

class LinearLayer: public Layer {
    public:
        // constructor
        LinearLayer(int rows, int cols): 
            Layer(std::make_shared<LinearTransformation>(rows, cols)) {}
            
        // destructor
        // TODO: always needed?
        ~LinearLayer() {};

        // initialize weights
        void Initialize(std::string initialization_type);

        // update weights and bias
        void UpdateWeights();
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

};

#endif // LAYER_H_