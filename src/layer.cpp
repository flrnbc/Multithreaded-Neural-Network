
#include <iostream>
#include <stdexcept>
#include "layer_cache.h"
#include "transformation.h"
#include "layer.h"


/*********
 * LAYER *
 *********/

void Layer::Input(std::vector<double> input_vector) {
        _layer_cache->SetForwardInput(std::make_shared<std::vector<double> >(input_vector));
}

std::vector<double> Layer::Output() {
        if (_layer_cache->GetForwardOutput() != nullptr) {
                return *(_layer_cache->GetForwardOutput());
        } else {
                throw std::invalid_argument("Output pointer is null!");
        }
}

void Layer::SetLayerCache(std::unique_ptr<LayerCache> layer_cache) {
        // TODO: this seems to work even though we did not define a move (assignment) operator
        // does it implicitly use the std::vector move (assignment) operator?
        _layer_cache = std::move(layer_cache);
}

/*******************************
 * LINEAR LAYER IMPLEMENTATION *
 *******************************/

void LinearLayer::Forward() {
        // NOTE: for 'connected' layers we have _forward_output != nullptr by the initialization
        // of a (sequential) neural network. Moreover, two layers share _forward_input and _forward_output 
        // respectively. We do not want to break this connection by setting the shared_ptr via 
        // std::make_shared<std::vector<double> > ... (which creates a new shared_ptr).
        // Instead we simply change the object owned by the shared_ptr.

        // TODO: need to include this function into the Transformation class since we use it over and over again.

        if (GetLayerCache()->GetForwardInput() != nullptr) {
                // TODO: do we copy the transformed vector too often?
                std::vector<double> transformed_vector = _transformation->Transform(*(GetLayerCache()->GetForwardInput()));
                if (GetLayerCache()->GetBackwardOutput() == nullptr){
                        GetLayerCache()->SetForwardOutput(std::make_shared<std::vector<double> >(transformed_vector));
                } else {
                        // TODO: do we create an unecessary copy of _backward_output here
                        *(GetLayerCache()->GetBackwardOutput()) = transformed_vector;
                }
        } else {
                throw std::invalid_argument("Pointer is null!");
        }          
}

void LinearLayer::Initialize(std::string initialization_type) {
        _transformation->Initialize(initialization_type);
        //->Initialize(initialization_type);
}


/********************
 * ACTIVATION LAYER *
 ********************/

void ActivationLayer::Forward() {
        if (GetLayerCache()->GetForwardInput() != nullptr) {
                // TODO: do we copy the transformed vector too often?
                std::vector<double> transformed_vector = _transformation->Transform(*(GetLayerCache()->GetForwardInput()));
                if (GetLayerCache()->GetBackwardOutput() == nullptr){
                        GetLayerCache()->SetForwardOutput(std::make_shared<std::vector<double> >(transformed_vector));
                } else {
                        // TODO: do we create an unecessary copy of _backward_output here
                        *(GetLayerCache()->GetBackwardOutput()) = transformed_vector;
                }
        } else {
                throw std::invalid_argument("Pointer is null!");
        }          
}


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




