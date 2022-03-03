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
        * ~Derivative~: updates the derivative of the transformation at a data point
        * ~BackwardTransform~: used for backward pass/backpropagation (TODO: better suited in Layer class?)
        * ~Summary~: summarizes most important parameters (TODO: too verbose at the moment)
    
    The following methods have an _empty_ implementation by default and do not have to be implemented by derived concrete classes:
        * ~UpdateWeightsBias~: only needed for linear transformations (TODO: could be replaced with 
          a more general member function e.g. ~UpdateParameters~)

    The concrete implementations of the abstract base class are:
        * ~LinearTransforamtion~: encapsulates an affine-linear transformation with weights and biases
        * ~ActivationTransformation~: could be a vectorized activation function or others, e.g. softmax.
    
    TODO: - ~BackwardTransform~ could be unified by including the derivative into the ~Transformation~ class.
*/

/***********************
 * ABSTRACT BASE CLASS *
 ***********************/

class Transformation {
    protected: // to make access easier for concrete derived classes
        // output dimension of a single data sample = number of rows (in case of a linear transformation/matrix)
        int _rows;
        // input dimension of a single data sample = number of columns (in case of a linear transformation/matrix)
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
        virtual Eigen::MatrixXd Transform(Eigen::MatrixXd) = 0;
        
        // 'backward transformation' for backpropagation; depends on the derivative
        // of the transformation at a given point (which is the first argument)
        virtual Eigen::RowVectorXd BackwardTransform(Eigen::VectorXd, Eigen::RowVectorXd) = 0;

        // compute derivative/jacobian of the transformation at the given vector
        virtual void Derivative(Eigen::VectorXd) = 0;

        // initialize transformation
        virtual void Initialize(std::string initialize_type="") {}

        // update weights/bias (only non-trivial for LinearTransformations)
        virtual void AddToWeights(Eigen::MatrixXd Delta_weights) {}
        virtual void AddToBias(Eigen::VectorXd Delta_bias) {}

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
        void Initialize(std::string) override;

        // transpose weight matrix
        void Transpose();

        // transform methods (just right matrix multiplication with input (a column vector))
        Eigen::MatrixXd Transform(Eigen::MatrixXd) override; 

        // get derivative (trivial here because it coincides with weights)
        void Derivative(Eigen::VectorXd) override {}

        // no dependence on the given point because the same applies to the derivative of an affine-linear transformation
        Eigen::RowVectorXd BackwardTransform(Eigen::VectorXd point, Eigen::RowVectorXd rowVector) override {
            return rowVector*(_weights);
        }

        // update weights/bias
        void AddToWeights(Eigen::MatrixXd deltaWeights) override {
            _weights += deltaWeights;
        }
        void AddToBias(Eigen::VectorXd deltaBias) override {
            _bias += deltaBias;
        }

        // summary
        std::string Summary() override;
};


/******************************************************
 * CONCRETE IMPLEMENTATION: ACTIVATION TRANSFORMATION *
 ******************************************************/

class ActivationTransformation: public Transformation {
    private:
        Function _function; // TODO: better to use a smart pointer?
        std::unique_ptr<Eigen::MatrixXd> _derivative; // derivative at a single vector (not at all data samples in a batch)

    public:
        // constructor
        ActivationTransformation(int size, std::string fct_name): 
            Transformation(size, size, fct_name),
            _function(Function(fct_name)),
            _derivative(std::make_unique<Eigen::MatrixXd>(Eigen::MatrixXd::Zero(size, size))) 
            {} 
            
        // transform method
        Eigen::MatrixXd Transform(Eigen::MatrixXd) override;

        // set derivative at the given point
        void Derivative(Eigen::VectorXd) override;

        // backward transformation for backpropagation; depends on the given point where
        // we take the derivative of the activation transformation
        Eigen::RowVectorXd BackwardTransform(Eigen::VectorXd input, Eigen::RowVectorXd input_matrix) override {
            Derivative(input);
            return input_matrix*(*(_derivative));
        }

        // Summary of transformation
        std::string Summary() override; 
};

#endif // TRANSFORMATION_H_