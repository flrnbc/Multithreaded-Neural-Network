#ifndef TRANSFORMATION_H_
#define TRANSFORMATION_H_

#include <cmath>
#include <Eigen/Dense>
#include "function.h"
#include <memory>
#include <vector>
#include <string>

/******************
 * TRANSFORMATION *
 ******************/

/** 
    Class for transformations used in layers of neural networks. It not only keeps track
    of the parameters for the transformation but also of its derivative (similar to Tensors
    in PyTorch).
    
    The abstract base class has three attributes:
        * cols (dimension of input)
        * rows (dimension of output)
        * type of transformation.
    By using an abstract base class, we can easily add further concrete implementations.
        
    The most important member functions, which need to be implemented by derived concrete classes, are:
        * ~Transform~: transforms the input, typically a vector or matrix 
          (NOTE: we work with _column vectors_ as our input. This might be a bit unconvential because often row vectors are used.)
        * ~Initialize~: initializes the parameters of the transformation (so far only for LinearTransformation)
        * ~UpdateDerivative~: updates the derivative of the transformation at a data point
        * ~UpdateDelta~: used for backward pass/backpropagation (TODO: better suited in Layer class?)
        * ~Summary~: summarizes most important parameters (TODO: too verbose at the moment)
    
    The following methods have an empty implementation by default and do not have to be implemented by derived concrete classes:
        * ~UpdateWeights~ / ~UpdateBias~: only needed for linear transformations (TODO: could be replaced with 
          a more general member function e.g. ~UpdateParameters~)

    The concrete implementations of the abstract base class are:
        * ~LinearTransforamtion~: encapsulates an affine-linear transformation with weights and biases
        * ~ActivationTransformation~: could be a vectorized activation function or others, e.g. softmax.
    
*/

/***********************
 * ABSTRACT BASE CLASS *
 ***********************/

class Transformation {
    protected:
        // output dimension
        int _rows;
        // input dimension   
        int _cols;
        // type of transformation
        std::string _type;

        // constructor (needed for concrete implementations to initialize the above member variables)
        Transformation(int rows, int cols, std::string type):
            _rows(rows),
            _cols(cols),
            _type(type)
            {}

    public:
        // virtual destructor
        virtual ~Transformation() {}

        // getters 
        // NOTE: no setters because the attributes are not supposed to be changed later on
        int Cols() { return _cols; }
        int Rows() { return _rows; }
        std::string Type() { return _type; }
        
        // TODO: can the following be improved using references?
        virtual Eigen::VectorXd Transform(Eigen::VectorXd) = 0;
        
        // update Delta (of a layer), which is a row vector, by right multiplication with the derivative
        // TODO: might move to Layer class
        virtual Eigen::RowVectorXd UpdateDelta(Eigen::RowVectorXd) = 0;

        // update derivative/jacobian of the transformation at a given vector
        virtual void UpdateDerivative(Eigen::VectorXd) = 0;

        // initialize transformation
        virtual void Initialize(std::string initialize_type="") {}

        // update weights/bias (only needed for LinearTransformations)
        virtual void UpdateWeights(Eigen::MatrixXd Delta_weights) {}
        virtual void UpdateBias(Eigen::VectorXd Delta_bias) {}

        // summary of transformation
        virtual std::string Summary() = 0;
};


/**************************************************
 * CONCRETE IMPLEMENTATION: LINEAR TRANSFORMATION *
 **************************************************/

class LinearTransformation: public Transformation {
    private:    
        Eigen::MatrixXd _weights;
        Eigen::VectorXd _bias;

    public:
        // constructors
        // constructor for (affine) linear transformations
        LinearTransformation(Eigen::MatrixXd weights, Eigen::VectorXd bias): 
            Transformation(weights.rows(), weights.cols(), "LinearTransformation"),
            _weights(weights),
            _bias(bias)
            {}

        // constructor (all zeros)
        LinearTransformation(int rows, int cols): 
            Transformation(rows, cols, "LinearTransformation"),
            _weights(Eigen::MatrixXd::Zero(rows, cols)), 
            _bias(Eigen::VectorXd::Zero(rows))
            {}

        // setters & getters
        Eigen::MatrixXd& Weights() { return _weights; }
        void SetWeights(Eigen::MatrixXd weight_matrix) { _weights = weight_matrix; }
        Eigen::VectorXd& Bias() { return _bias; }
        void SetBias(Eigen::VectorXd bias) { _bias = bias; }

        // random initialize (either via "He" or "Normalized Xavier")
        void Initialize(std::string);

        // transpose weight matrix
        void Transpose();

        // transform methods (just right matrix multiplication with input (a column vector))
        Eigen::VectorXd Transform(Eigen::VectorXd); 

        // update derivative (trivial here because it coincides with weights)
        void UpdateDerivative(Eigen::VectorXd) {}

        // update Delta
        // used to compute the gradient of a loss function successively via backpropagation
        Eigen::RowVectorXd UpdateDelta(Eigen::RowVectorXd rowVector) {
            return rowVector*(_weights);
        }

        // update weights/bias
        void UpdateWeights(Eigen::MatrixXd deltaWeights) {
            _weights += deltaWeights;
        }
        void UpdateBias(Eigen::VectorXd deltaBias) {
            _bias += deltaBias;
        }

        // summary
        std::string Summary();
};


/******************************************************
 * CONCRETE IMPLEMENTATION: ACTIVATION TRANSFORMATION *
 ******************************************************/

class ActivationTransformation: public Transformation {
    private:
        Function _function; // TODO: better to use a smart pointer?
        std::unique_ptr<Eigen::MatrixXd> _derivative;

    public:
        // constructor
        ActivationTransformation(int size, std::string fct_name): 
            Transformation(size, size, fct_name),
            _function(Function(fct_name)),
            _derivative(std::make_unique<Eigen::MatrixXd>(Eigen::MatrixXd::Zero(size, size))) 
            {} 
            
        // transform methods
        Eigen::VectorXd Transform(Eigen::VectorXd);

        // update derivative at a point/vector (trivial here)
        void UpdateDerivative(Eigen::VectorXd);

        // update Delta
        Eigen::RowVectorXd UpdateDelta(Eigen::RowVectorXd rowVector) {
            return rowVector*(*(_derivative));
        }

        // Summary of transformation
        std::string Summary(); 
};


/** Possible future transformations:
        * Flatten layer

*/

#endif // TRANSFORMATION_H_