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
        * ~_transformation~: shared pointer to the Transformation of the corresponding layer,
        * ~_layer_cache~: shared pointer to a LayerCache to store the data of the forward and backward pass.

    The most important member functions are
        * ~Input~: sets the vector in forward input of the LayerCache
        * ~Forward~: applies the transformation to the input and stores it in forward output
        * ~Output~: returns the vector in forward output
        
        * ~BackwardInput~, ~Backward~ and ~BackwardOutput: work analogously as ~Input~, ~Forward~ and ~Output~
          only with the derivative of the transformation
    
    NOTE: some of the member variables, e.g. ~Cols~, ~Rows~, ~UpdateWeightsBias~, just simplify the access
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

        // used for updating the weights/bias (empty for ActivationLayer)
        virtual void UpdateDeltaWeights(Eigen::VectorXd, Eigen::RowVectorXd) {}
        virtual void UpdateDeltaBias(Eigen::RowVectorXd) {}
        virtual void ZeroDeltaWeights() {}
        virtual void ZeroDeltaBias() {}

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
        void Input(Eigen::MatrixXd); // TODO: do we copy too often here?
        void Forward();
        Eigen::MatrixXd Output(); // TODO: better to return const ref?

        // backward pass
        //void Derivative();
        void BackwardInput(Eigen::MatrixXd);
        void Backward();
        Eigen::MatrixXd BackwardOutput();

        // update weights/bias (empty for ActivationLayers)
        virtual void UpdateWeightsBias(double learning_rate) {}
};


/**********************
 * LINEAR LAYER CLASS *
 **********************/
 
// NOTE: possible IDEA for refactoring: use a factory pattern for creating various layers?

class LinearLayer: public Layer {
    protected:
        // Delta of weights/bias used in backpropagation to update weights/bias
        Eigen::MatrixXd _DeltaWeights;
        Eigen::VectorXd _DeltaBias;

        // update Delta of weights/bias with the LayerCache
        virtual void UpdateDeltaWeights(Eigen::VectorXd, Eigen::RowVectorXd) override;
        virtual void UpdateDeltaBias(Eigen::RowVectorXd) override;
        // set Delta of weights/bias to zero
        virtual void ZeroDeltaWeights() override;
        virtual void ZeroDeltaBias() override;

    public:
        // constructors
        LinearLayer(Eigen::MatrixXd weights, Eigen::VectorXd bias):
            Layer(std::make_shared<LinearTransformation>(weights, bias)),
            _DeltaWeights(Eigen::MatrixXd::Zero(weights.rows(), weights.cols())),
            _DeltaBias(Eigen::VectorXd::Zero(bias.rows()))
             {}
        LinearLayer(int rows, int cols): 
            Layer(std::make_shared<LinearTransformation>(rows, cols)), 
            _DeltaWeights(Eigen::MatrixXd::Zero(rows, cols)),
            _DeltaBias(Eigen::VectorXd::Zero(rows))
            {}

        // initialize weights
        void Initialize(std::string initialization_type);

        // update weights and bias
        void UpdateWeightsBias(double learning_rate) override;
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