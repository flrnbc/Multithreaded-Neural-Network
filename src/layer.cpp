
#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>
#include "layer_cache.h"
#include "transformation.h"
#include "layer.h"


/*********
 * LAYER *
 *********/

void Layer::SetLayerCache(std::unique_ptr<LayerCache> layer_cache) {
        // TODO: this seems to work even though we did not define a move (assignment) operator
        // does it implicitly use the std::vector move (assignment) operator?
        _layer_cache = std::move(layer_cache);
}

// TODO #A: setting the in-/output of for-/backward pass needs to be refactored at some point

// in-/output for forward pass
void Layer::Input(Eigen::VectorXd input_vector) {
        _layer_cache->SetForwardInput(std::make_shared<Eigen::VectorXd>(input_vector));
}

Eigen::VectorXd Layer::Output() {
        if (_layer_cache->GetForwardOutput() == nullptr) {
                throw std::invalid_argument("Forward output pointer is null!");       
        } 
        return *(_layer_cache->GetForwardOutput());
}

// forward pass
void Layer::Forward() {
        // NOTE: for 'connected' layers we have _forward_output != nullptr by the initialization
        // of a (sequential) neural network. Moreover, two layers share _forward_input and _forward_output 
        // respectively. We do not want to break this connection by setting the shared_ptr via 
        // std::make_shared<Eigen::VectorXd > ... (which creates a new shared_ptr).
        // Instead we simply change the object owned by the shared_ptr.

        // TODO: need to include this function into the Transformation class since we use it over and over again.

        if (GetLayerCache().GetForwardInput() != nullptr) {
                // TODO: do we copy the transformed vector too often?
                Eigen::VectorXd transformed_vector = _transformation->Transform(*(GetLayerCache().GetForwardInput()));
                if (GetLayerCache().GetForwardOutput() == nullptr){
                        GetLayerCache().SetForwardOutput(std::make_shared<Eigen::VectorXd>(transformed_vector));
                } else {
                        // TODO: do we create an unecessary copy here?
                        *(GetLayerCache().GetForwardOutput()) = transformed_vector;
                }
        } else {
                throw std::invalid_argument("Pointer is null!");
        }          
}

// update derivative for backward pass
void Layer::UpdateDerivative() {
        if (_layer_cache->GetForwardOutput() == nullptr) {
                throw std::invalid_argument("Forward output pointer is null!");   
        } 
        _transformation->UpdateDerivative(*(_layer_cache->GetForwardOutput()));
}


// in-/output for backward pass
void Layer::BackwardInput(Eigen::RowVectorXd input_vector) {
        _layer_cache->SetBackwardInput(std::make_shared<Eigen::RowVectorXd>(input_vector));
}

Eigen::RowVectorXd Layer::BackwardOutput() {
        if (_layer_cache->GetBackwardOutput() != nullptr) {
                return *(_layer_cache->GetBackwardOutput());
        } else {
                throw std::invalid_argument("Output pointer is null!");
        }
}




// TODO #A: Both Forward() and Backward() need to be refactored at some point

/*******************************
 * LINEAR LAYER IMPLEMENTATION *
 *******************************/

void LinearLayer::Initialize(std::string initialization_type) {
        _transformation->Initialize(initialization_type);
        //->Initialize(initialization_type);
}

void LinearLayer::Backward() {
        // get backward input which is the Delta of the previous layer of the backward pass
        if (GetLayerCache().GetBackwardInput() != nullptr) {
                // TODO: do we copy the transformed vector too often?
                Eigen::RowVectorXd transformed_vector = _transformation->UpdateDelta(*(GetLayerCache().GetBackwardInput()));
                if (GetLayerCache().GetBackwardOutput() == nullptr){
                        GetLayerCache().SetBackwardOutput(std::make_shared<Eigen::RowVectorXd>(transformed_vector));
                } else {
                        // TODO: do we create an unecessary copy here
                        *(GetLayerCache().GetBackwardOutput()) = transformed_vector;
                }
        } else {
                throw std::invalid_argument("Pointer is null!");
        }          
}


/********************
 * ACTIVATION LAYER *
 ********************/

void ActivationLayer::Backward() {
        // get backward input which is the Delta of the previous layer of the backward pass
        if (GetLayerCache().GetBackwardInput() != nullptr) {
                // TODO: do we copy the transformed vector too often?
                Eigen::RowVectorXd transformed_vector = _transformation->UpdateDelta(*(GetLayerCache().GetBackwardInput()));
                if (GetLayerCache().GetBackwardOutput() == nullptr){
                        GetLayerCache().SetBackwardOutput(std::make_shared<Eigen::RowVectorXd>(transformed_vector));
                } else {
                        // TODO: do we create an unecessary copy here
                        *(GetLayerCache().GetBackwardOutput()) = transformed_vector;
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




