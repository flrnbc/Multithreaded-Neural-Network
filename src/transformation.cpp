#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "transformation.h"

/* 
    Helper functions
*/
// helper function to print double vectors
std::string Transformation::PrintDoubleVector(const std::vector<double>& double_vector) {
    std::string vector_string = "";
    for (double d: double_vector) {
        vector_string += std::to_string(d) + ",\t";
    }
    return vector_string;
}

// helper function to transpose
// TODO: improve with move semantics?!
static std::vector<std::vector<double> > Transpose(const std::vector<std::vector<double> >& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<double> > transposedMatrix(rows, std::vector<double>(cols, 0));

    for (int i=0; i < matrix.size(); i++) {
        for (int j=0; j < matrix[0].size(); j++) {
            transposedMatrix[i][j] = matrix[j][i];
        }
    }

    return transposedMatrix;
}

/*
    LinearTransformation
*/
// transform methods
std::vector<double> LinearTransformation::Transform(std::vector<double> inputVector) {
    if (inputVector.size() != Cols()) {
        throw std::domain_error("Vector cannot be evaluated.");
    } else {
        // TODO #A: really need to copy the weights?
        std::vector<std::vector<double> > weights = this->Weights();
        std::vector<double> outputVector(weights.size(), 0);
        std::vector<double> bias = this->Bias();

        for (int i=0; i < this->Rows(); i++) {
            // matrix multiplication weights*inputVector
            for (int j=0; j < this->Rows(); j++) {
                outputVector[i] += weights[i][j]*inputVector[j];
            }
            // add bias 
            outputVector[i] += bias[i];
        }
        return outputVector;
    }
}

std::vector<std::vector<double> > LinearTransformation::Transform(std::vector<std::vector<double> > inputMatrix) {
    // rows of inputMatrix = inputMatrix[0].size()
    // cols of inputMatrix = inputMatrix.size()
    if ((inputMatrix[0].size() != Cols()) || (inputMatrix.size() != Rows())) {
        throw std::domain_error("Matrices cannot be multiplied.");
    } else {
        // TODO #A: really need to copy the weights?
        std::vector<std::vector<double> > weights = this->Weights();
        // TODO #A: again might be much better to use move semantics
        // NOTE: the trick here is to use the transpose so that we can access the columns vectors of inputMatrix
        // more easily and use the previous method. Of course, we then have to apply the transpose again to get 
        // the correct output.
        std::vector<std::vector<double> > transposedInput = Transpose(inputMatrix);
        std::vector<std::vector<double> > transposedOutputMatrix(Cols(), std::vector<double>(Rows(), 0));

        // TODO #A: check if above idea works in detail!
        for (int i=0; i < Rows(); i++) {
            transposedOutputMatrix[i] = Transform(transposedInput[i]);
        }
        return Transpose(transposedOutputMatrix);
    }
}

// Summary
// TODO #A: extend a bit more
std::string LinearTransformation::Summary() {
    std::string rows = std::to_string(Rows());
    std::string cols = std::to_string(Cols());
    std::string shape = "Shape: (" + rows + ", " + cols + ")" + "\n";

    std::string weights_string = "Weights:\n";
    for (int i=0; i < this->Rows(); i++) {
        weights_string += Transformation::PrintDoubleVector(this->Weights()[i]) + "\n";
    }

    std::string bias_string = "Bias:\n" + Transformation::PrintDoubleVector(this->Bias());

    return shape + weights_string + bias_string;
}



/* 
    Activation functions
*/
double heaviside(double x) {
    if (x < 0) { return 0; }
    return 1;
}

double identity(double x) {
    return x;
}

double prelu(double x, double a) {
    if (x <= 0) { return a*x; }
    return x;
}

double relu(double x) {
    return prelu(x, 0);
}

double sigmoid(double x) {
    return 1/(1 + exp(-x));
}

/*
    Constructor for activation function
*/
ActivationTransformation::ActivationTransformation(int size, std::string fct_name="identity") {
    SetCols(size);
    SetRows(size);
    if (fct_name == "heaviside") {
        activation_fct = &heaviside;
    }
    else if (fct_name == "identity") {
        activation_fct = &identity;
    }
    // TODO #B: to simplify, only offer relu and not prelu at the moment
    else if (fct_name == "relu") {
        activation_fct = &relu;
    }
    else if (fct_name == "sigmoid") {
        activation_fct = &sigmoid;
    }
    else if (fct_name == "tanh") {
        activation_fct = &tanh;
    }
    else throw std::invalid_argument("Not a valid activation function.");

    // finally set the name
    this->SetName(fct_name);
}


/*
    Implementing the transform methods for ActivationTransformation
*/
std::vector<double> ActivationTransformation::Transform(std::vector<double> vector) {
    // TODO #A: optimize vectorization! (OpenMP?)
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = activation_fct(vector[i]); // NOTE: range-based did not modify the values (even when using &)?!
    }

    return vector;
}

std::vector<std::vector<double> > ActivationTransformation::Transform(std::vector<std::vector<double> > matrix) {
    for (int i=0; i < matrix.size(); i++) {
        matrix[i] = Transform(matrix[i]);
    }
    
    return matrix;
}

std::string ActivationTransformation::Summary() {
    std::string rows = std::to_string(Rows());
    std::string cols = std::to_string(Cols());
    std::string shape = "Shape: (" + rows + ", " + cols + ")" + "\n";

    return shape + "Function name: " + Name();
}