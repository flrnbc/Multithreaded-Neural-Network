#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "transformation.h"


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
        // TODO #A: again might be better to use move semantics
        // NOTE: the trick here is to use the transpose so that we can access the columns vectors of inputMatrix
        // more easily and use the previous method
        std::vector<std::vector<double> > transposedInput = Transpose(inputMatrix);
        std::vector<std::vector<double> > transposedOutputMatrix(Cols(), std::vector<double>(Rows(), 0));

        for (int i=0; i < Rows(); i++) {
            transposedOutputMatrix[i] = Transform(transposedInput[i]);
        }
        return Transpose(transposedOutputMatrix);
    }
}