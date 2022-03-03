#include "transformation.h"
#include <Eigen/Dense>

#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

/******************
 * TRANSFORMATION *
 ******************/

/**************************************
 * HELPER FUNCTION FOR RANDOM NUMBERS *
 **************************************/
// TODO: is there a performance issue? (since we create a generator etc. each time)?
double RandomNumberUniform(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);

    return dis(gen);
}

double RandomNumberNormal(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(min, max);

    return dis(gen);
}


/*************************
 * LINEAR TRANSFORMATION *
 *************************/

// random initialization (according to the previous activation layer)
// either He or normalized Xavier initialization
void LinearTransformation::Initialize(std::string initialization_type) {
    if (initialization_type != "He" && initialization_type != "Xavier") {
        throw std::domain_error("Initialization type is unknown.");
    }

    int rows = Rows();
    int cols = Cols();

    // actual initialization of weights
    if (initialization_type == "Xavier") {      
        // use normalized Xavier weight initialization
        int inputPlusOutputSize = cols + rows;
        // uniform_real_distribution(a, b) generates for a real number in the half-open interval [a, b)
        double min = -(sqrt(6)/sqrt(inputPlusOutputSize));
        double max = sqrt(6)/sqrt(inputPlusOutputSize);

        // randomly initialize weights
        // NOTE: be careful with cache-friendliness (Eigen uses 'column major' storage by default, so outer loop over columns
        // for cache-friendliness)
        for (int j=0; j<cols; j++) {
            for (int i=0; i<rows; i++) {
                _weights(i, j) = RandomNumberUniform(min, max); 
            }
        }
    } 
    else {
        // use He weight initialization
        double min = 0.0;
        double max = sqrt(2.0/cols);

        // randomly initialize weights
        for (int j=0; j<cols; j++) {
            for (int i=0; i<rows; i++) {
                _weights(i, j) = RandomNumberNormal(min, max); 
            }
        }
    }
}

// transform method
Eigen::MatrixXd LinearTransformation::Transform(Eigen::MatrixXd inputMatrix) {
    if (inputMatrix.rows() != Cols()) {
        throw std::domain_error("Matrix cannot be evaluated.");
    } else {
        Eigen::MatrixXd result = _weights*inputMatrix;
        // apply broadcasting for adding the bias (add _bias to each column of result)
        result.colwise() += _bias;
        return result;
    }
}

// Summary
std::string LinearTransformation::Summary() {
    std::string type = Type() + "\n";
    std::string rows = std::to_string(Rows());
    std::string cols = std::to_string(Cols());
    std::string shape = "Shape: (" + rows + ", " + cols + ")" + "\n";

    std::string weights_string = "Weights:\n";
    // convert matrix to string
    // TODO: at the moment a bit verbose (matrices might be large...)
    std::stringstream ss_weights;
    ss_weights << _weights;
    weights_string += ss_weights.str();

    std::string bias_string = "Bias:\n";
    // convert vector to string
    std::stringstream ss_bias;
    ss_bias << _bias;
    bias_string += ss_bias.str();

    return type + shape + weights_string + "\n" + bias_string;
}


/*****************************
 * ACTIVATION TRANSFORMATION *
 *****************************/

// transform methods for ActivationTransformation
Eigen::MatrixXd ActivationTransformation::Transform(Eigen::MatrixXd inputMatrix) {
    Eigen::MatrixXd outputMatrix = Eigen::MatrixXd::Zero(inputMatrix.rows(), inputMatrix.cols());
    // correction term, see below
    double correction = 0.0;

    // TODO: optimize with Eigen's  unaryExpr?
    // NOTE: Eigen uses 'column major' storage by default so that this is cache-friendly
    for (int j=0; j < inputMatrix.cols(); j++) {
        for (int i = 0; i < inputMatrix.rows(); i++) {
            if (_type == "softmax") {
                // substracting the max coefficient for numerical stability (see https://cs231n.github.io/linear-classify/#softmax-classifier)
                // from each column
                correction = inputMatrix.col(j).maxCoeff();
            }
            outputMatrix(i, j) = _function(inputMatrix(i, j)-correction); 
        }
    }

    if (_type == "softmax") {
        double colSum = 0;
        for (int j=0; j<outputMatrix.cols(); j++) {
            colSum = outputMatrix.col(j).sum();
            for (int i = 0; i < outputMatrix.rows(); i++) {
                outputMatrix(i, j) /= colSum; // Note: this is safe because _function = exp for softmax
            }
        }
    }

    return outputMatrix;
}

// update derivative
void ActivationTransformation::Derivative(Eigen::VectorXd vector) {
    if (vector.size() != Cols()) {
        throw std::domain_error("Vector size does not match with the derivative.");
    }

    if (_type != "softmax") {
        for (int j=0; j<Cols(); j++) {
            // safe because we initialize to a zero matrix
            (*_derivative)(j, j) = _function.Derivative(vector(j));
        }
    }

    if (_type == "softmax") {
        // TODO: it would be more efficient to retrieve this result from the forward output of the corresponding LayerCache
        Eigen::VectorXd softmax = this->Transform(vector); 

        for (int j=0; j<Cols(); j++) {
            for (int i=0; i<Rows(); i++) {
                if (i == j) {
                    (*_derivative)(i, i) = softmax(i)*(1-softmax(i));
                }
                else {
                    (*_derivative)(i, j) = -softmax(i)*softmax(j);
                }
            }
        }
    }
}

std::string ActivationTransformation::Summary() {
    std::string rows = std::to_string(Rows());
    std::string cols = std::to_string(Cols());
    std::string shape = "Shape: (" + rows + ", " + cols + ")" + "\n";

    std::string derivative_string = "Derivative:\n";
    // convert matrix to string
    std::stringstream ss_derivative;
    ss_derivative << *_derivative;
    derivative_string += ss_derivative.str();

    return shape + "Function name: " + Type() + "\n" + derivative_string;
}