
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "layer.h"
#include "transformation.h"


// setters/getters
void Layer::SetForwardInput(std::shared_ptr<std::vector<double> > input_ptr) {
       _forward_input = std::move(input_ptr);
}

void Layer::SetForwardOutput(std::shared_ptr<std::vector<double> > output_ptr) {
        _forward_output = std::move(output_ptr);
}

std::shared_ptr<std::vector<double> > Layer::GetForwardOutput() {
        return _forward_output;
}

std::shared_ptr<std::vector<double> > Layer::GetForwardInput() {
        return _forward_input;
}

void Layer::SetBackwardInput(std::shared_ptr<std::vector<double> > backward_input_ptr) {
        _backward_input = std::move(backward_input_ptr);
}

std::shared_ptr<std::vector<double> > Layer::GetBackwardOutput() { 
        return _backward_output;
}

void LinearLayer::Forward() {
        // NOTE: for 'connected' layers we have _forward_output != nullptr by the initialization
        // of a (sequential) neural network. Moreover, two layers share _forward_input and _forward_output 
        // respectively. We do not want to break this connection by setting the shared_ptr via 
        // std::make_shared<std::vector<double> > ... (which creates a new shared_ptr).
        // Instead we simply change the object owned by the shared_ptr.

        if (GetForwardInput() != nullptr) {
                // TODO: do we copy the transformed vector too often?
                std::vector<double> transformed_vector = _transformation->Transform(*(GetForwardInput()));
                if (GetBackwardOutput() == nullptr){
                        SetForwardOutput(std::make_shared<std::vector<double> >(transformed_vector));
                } else {
                        // TODO: do we create an unecessary copy of _backward_output here
                        *(GetBackwardOutput()) = transformed_vector;
                }
        } else {
                throw std::invalid_argument("Pointer is null!");
        }          
}


/*******************************
 * LINEAR LAYER IMPLEMENTATION *
 *******************************/

LinearLayer::LinearLayer(int rows, int cols) {
        _transformation = std::make_unique<LinearTransformation>(rows, cols); 

}

void LinearLayer::Initialize(std::string initialization_type) {
        _transformation->Initialize(initialization_type);
}

std::string LinearLayer::Summary() {
        return _transformation->Summary();
}


// Layer //
// constructor for Layer
// Layer::Layer(int rows, int cols, std::string activation) : 
//         LayerBase(rows, cols, activation), 
//         _next(nullptr),
//         _previous(nullptr) {}

// // setters & getters for Layer
// void Layer::SetNext(std::shared_ptr<Layer> next) {
//         _next = next; 
// }

// void Layer::SetPrevious(std::shared_ptr<Layer> previous) { 
//         _previous = previous;
// }

// // flatten layer
// std::vector<double> Layer::Flatten(const std::vector<std::vector<double> >& matrix) {
//         int rows = matrix.size();
//         int cols = matrix[0].size();
//         int output_size = rows*cols;

//         std::vector<double> output(output_size, 0);

//         for (int i=0; i<rows; i++) {
//                 for (int j=0; j<cols; j++) {
//                         output[i+j] = matrix[i][j];
//                 }
//         }

//         return output;
// }


// // forward pass
// // NOTE: no need to update for input layer (need to set input though)
// void Layer::UpdateInput() {
//         if (Previous() != nullptr) {
//                 std::cout << "from UpdateInput: " << Perceptron::PrintDoubleVector(Previous()->OutputData()) << std::endl;
//                 SetInputData(Previous()->OutputData());
//         }
// }

// void LayerBase::UpdateOutput() {
//         SetOutputData(Perceptron()->Evaluate(InputData()));
// }

// void Layer::Forward() {
//         UpdateInput();
//         UpdateOutput();
// }

// backward pass
// TODO #A: ADD!!!

