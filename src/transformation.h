#ifndef TRANSFORMATION_H_
#define TRANSFORMATION_H_

#include <cmath>
#include <Eigen/Dense>
#include "function.h"
#include <memory>
#include <vector>
#include <string>

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
        
        // transform method (want to keep input, so no pass by reference)
        virtual Eigen::VectorXd Transform(Eigen::VectorXd) = 0;
        //std::vector<std::vector<double> > Transform(std::vector<std::vector<double> >);
        
        // update Delta (of a layer), which is a row vector, by right multiplication with the derivative
        virtual Eigen::RowVectorXd UpdateDelta(Eigen::RowVectorXd) = 0;

        // update derivative/jacobian from a given vector
        virtual void UpdateDerivative(Eigen::VectorXd) = 0;

        // initialize transformation (many transformations have empty implementation)
        virtual void Initialize(std::string initialize_type="") {}
        
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

        // TODO: the default copy and move constructors/assignment operators should be enough?!?

        // setters & getters
        Eigen::MatrixXd Weights() { return _weights; }
        void SetWeights(Eigen::MatrixXd weight_matrix) { _weights = weight_matrix; }
        Eigen::VectorXd Bias() { return _bias; }
        void SetBias(Eigen::VectorXd bias) { _bias = bias; }

        // random initialize (either via "He" or "Normalized Xavier")
        void Initialize(std::string);

        // transpose weight matrix
        void Transpose();

        // transform methods (just right matrix multiplication with input)
        // NOTE: we do _not_ work with transpose etc. because we consider our data points as column vectors
        Eigen::VectorXd Transform(Eigen::VectorXd); 

        // update derivative (trivial here because it coincides with weights)
        void UpdateDerivative(Eigen::VectorXd) {}

        // update Delta
        Eigen::RowVectorXd UpdateDelta(Eigen::RowVectorXd rowVector) {
            return rowVector*(_weights);
        }

        // summaries
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

        // update derivative at a point/vector
        void UpdateDerivative(Eigen::VectorXd);

        // update Delta
        Eigen::RowVectorXd UpdateDelta(Eigen::RowVectorXd rowVector) {
            return rowVector*(*(_derivative));
        }

        // Summary of transformation
        std::string Summary(); 
};

// TODO: to include: Flatten layer

#endif // TRANSFORMATION_H_