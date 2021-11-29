#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "transformation.h"

/********************
 * HELPER FUNCTIONS *
 ********************/

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
std::vector<std::vector<double> > LinearTransformation::Transpose(const std::vector<std::vector<double> >& matrix) {
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

// helper function for random numbers
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


/******************
 * TRANSFORMATION *
 ******************/

// Transformation::Transformation() {
//     // not meant to be called directly
//     SetCols(0);
//     SetRows(0);
// }

// std::vector<double> Transformation::Transform(std::vector<double> input) {
//     // just the identity
//     return input;
// }

// std::string Transformation::Summary() {
//     return "Identity transformation.";
// }


/*************************
 * LINEAR TRANSFORMATION *
 *************************/

// constructor
LinearTransformation::LinearTransformation(std::vector<std::vector<double> > weights, std::vector<double> bias) {
    SetWeights(weights);
    SetBias(bias);
    SetRows(weights.size());
    SetCols(weights[0].size());
}

// constructor setting weights and bias to zero
LinearTransformation::LinearTransformation(int rows, int cols) {
    SetCols(cols);
    SetRows(rows);

    SetWeights(std::vector<std::vector<double> >(rows, std::vector<double>(cols, 0))); // implicitly uses move semantics?
    SetBias(std::vector<double>(rows, 0));
}

// random initialization (used for different activation functions)
// either He or normalized Xavier initialization
void LinearTransformation::Initialize(std::string initialization_type) {
    if (initialization_type != "He" && initialization_type != "Xavier") {
        // TODO #A: need to catch it somewhere?
        throw std::domain_error("Initialization type is unknown.");
    }

    int rows = Rows();
    int cols = Cols();

    // actual initialization of weights
    if (initialization_type == "Xavier") {      
        // use normalized Xavier weight initialization
        int inputPlusOutputSize = cols + rows;
        // NOTE: uniform_real_distribution(a, b) generates for [a, b) (half-open interval)
        double min = -(sqrt(6)/sqrt(inputPlusOutputSize));
        double max = sqrt(6)/sqrt(inputPlusOutputSize);

        // randomly initialize weights
        // NOTE: be careful with cache-friendliness (outer loop over rows)
        // TODO #A: would be nicer to put this for-loop after the else-block (but then would have to declare dis before; but as what?)
        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {
                _weights[i][j] = RandomNumberUniform(min, max); 
            }
        }
    } 
    else {
        // use He weight initialization
        double min = 0.0;
        double max = sqrt(2.0/cols);

        // randomly initialize weights
        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {
                _weights[i][j] = RandomNumberNormal(min, max); 
            }
        }
    }

}


// transform method
std::vector<double> LinearTransformation::Transform(std::vector<double> inputVector) {
    if (inputVector.size() != Cols()) {
        // TODO #A: catch exception somewhere?
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

// std::vector<std::vector<double> > LinearTransformation::Transform(std::vector<std::vector<double> > inputMatrix) {
//     // rows of inputMatrix = inputMatrix[0].size()
//     // cols of inputMatrix = inputMatrix.size()
//     if ((inputMatrix[0].size() != Cols()) || (inputMatrix.size() != Rows())) {
//         throw std::domain_error("Matrices cannot be multiplied.");
//     } else {
//         // TODO #A: really need to copy the weights?
//         std::vector<std::vector<double> > weights = this->Weights();
//         // TODO #A: again might be much better to use move semantics
//         // NOTE: the trick here is to use the transpose so that we can access the columns vectors of inputMatrix
//         // more easily and use the previous method. Of course, we then have to apply the transpose again to get 
//         // the correct output.
//         std::vector<std::vector<double> > transposedInput = LinearTransformation::Transpose(inputMatrix);
//         std::vector<std::vector<double> > transposedOutputMatrix(Cols(), std::vector<double>(Rows(), 0));

//         // TODO #A: check if above idea works in detail!
//         for (int i=0; i < Rows(); i++) {
//             transposedOutputMatrix[i] = Transform(transposedInput[i]);
//         }
//         return Transpose(transposedOutputMatrix);
//     }
// }

// Summary
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


/************************
 * ACTIVATION FUNCTIONS *
 ************************/

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

/*****************************
 * ACTIVATION TRANSFORMATION *
 *****************************/

ActivationTransformation::ActivationTransformation(int size, std::string fct_name) {
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


// transform methods for ActivationTransformation
std::vector<double> ActivationTransformation::Transform(std::vector<double> vector) {
    // TODO #A: optimize vectorization! (OpenMP?)
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = activation_fct(vector[i]); // NOTE: range-based did not modify the values (even when using &)?!
    }

    return vector;
}

// std::vector<std::vector<double> > ActivationTransformation::Transform(std::vector<std::vector<double> > matrix) {
//     for (int i=0; i < matrix.size(); i++) {
//         matrix[i] = Transform(matrix[i]);
//     }
    
//     return matrix;
// }

std::string ActivationTransformation::Summary() {
    std::string rows = std::to_string(Rows());
    std::string cols = std::to_string(Cols());
    std::string shape = "Shape: (" + rows + ", " + cols + ")" + "\n";

    return shape + "Function name: " + Name();
}