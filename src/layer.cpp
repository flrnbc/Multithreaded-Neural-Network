
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
        // NOTE: uses the move semantics of Eigen?!
        _layer_cache = std::move(layer_cache);
}

// TODO: setting the in-/output of for-/backward pass needs to be refactored at some point (e.g. moved to LayerCache)

// in-/output for forward pass
void Layer::Input(Eigen::MatrixXd input_vector) {
        _layer_cache->SetForwardInput(std::make_shared<Eigen::MatrixXd>(input_vector));
}

Eigen::MatrixXd Layer::Output() {
        if (_layer_cache->GetForwardOutput() == nullptr) {
                throw std::invalid_argument("Forward output pointer is null!");       
        } 
        return *(_layer_cache->GetForwardOutput());
}

// forward pass
void Layer::Forward() {
        /** NOTE: for 'connected' layers we have _forward_output != nullptr by the initialization
            of a (sequential) neural network. Moreover, two layers share _forward_input and _forward_output 
            respectively. We do not want to break this connection by setting the shared_ptr via 
            std::make_shared<Eigen::VectorXd > ... (which creates a new shared_ptr).
            Instead we simply change the object owned by the shared_ptr.
        */
        if (GetLayerCache().GetForwardInput() == nullptr) {
                throw std::invalid_argument("Pointer is null!");
        }
        // TODO: do we copy the transformed matrix too often?
        Eigen::MatrixXd transformed_matrix = _transformation->Transform(*(GetLayerCache().GetForwardInput()));
        if (GetLayerCache().GetForwardOutput() == nullptr){
                GetLayerCache().SetForwardOutput(std::make_shared<Eigen::MatrixXd>(transformed_matrix));
        } else {
                // TODO: do we create an unecessary copy here?
                *(GetLayerCache().GetForwardOutput()) = transformed_matrix;
        }
}


// in-/output for backward pass
void Layer::BackwardInput(Eigen::MatrixXd input_matrix) {
        _layer_cache->SetBackwardInput(std::make_shared<Eigen::MatrixXd>(input_matrix));
}

Eigen::MatrixXd Layer::BackwardOutput() {
        if (_layer_cache->GetBackwardOutput() == nullptr) {
                throw std::invalid_argument("Output pointer is null!");   
        }
        return *(_layer_cache->GetBackwardOutput());
}

// backward pass
void Layer::Backward() {
        // get backward input which is the Delta of the previous layer in the the backward pass (backpropagation)
        if ((GetLayerCache().GetBackwardInput() == nullptr) || (GetLayerCache().GetForwardInput() == nullptr)) {
                throw std::invalid_argument("Pointer to backward/forward input is null!");
        }
        Eigen::MatrixXd& backward_input = *(GetLayerCache().GetBackwardInput()); // number of rows = number of data samples
        Eigen::MatrixXd& forward_input = *(GetLayerCache().GetForwardInput()); 
        Eigen::MatrixXd transformed_matrix = Eigen::MatrixXd::Zero(backward_input.rows(), Cols());
        // backward transform/backpropagation of the i-th row of the backward_input (= propagated loss function gradient of 
        // the i-th sample) with respect to the derivative of the transformation at the i-th column of the forward_input
        for (int i=0; i<backward_input.rows(); i++) {
                transformed_matrix.row(i) = _transformation->BackwardTransform(forward_input.col(i), backward_input.row(i));
        }
        // set backard output (which might be a null pointer) to transformed_matrix
        if (GetLayerCache().GetBackwardOutput() == nullptr) {
                GetLayerCache().SetBackwardOutput(std::make_shared<Eigen::MatrixXd>(transformed_matrix));
        } else {
                // TODO: do we create an unecessary copy here
                *(GetLayerCache().GetBackwardOutput()) = transformed_matrix;
        }
}


/*******************************
 * LINEAR LAYER IMPLEMENTATION *
 *******************************/

void LinearLayer::Initialize(std::string initialization_type) {
        _transformation->Initialize(initialization_type);
        //->Initialize(initialization_type);
}

// set _DeltaWeights and _DeltaBias to zero
void LinearLayer::ZeroDeltaWeights() {
        int rows = GetTransformation()->Rows();
        int cols = GetTransformation()->Cols();
        _DeltaWeights = Eigen::MatrixXd::Zero(rows, cols);
}
void LinearLayer::ZeroDeltaBias() {
        int rows = GetTransformation()->Rows();
        _DeltaBias = Eigen::VectorXd::Zero(rows);
}

// update _DeltaWeights and _DeltaBias
// 'input' will be columns of the forward_input of this layer (see UpdateWeights etc.) and backward_input
// comes from backpropagation
void LinearLayer::UpdateDeltaWeights(Eigen::VectorXd input, Eigen::RowVectorXd backward_input) {
        // the next formula follows from the backpropagation algorithm
        _DeltaWeights += backward_input.transpose()*input.transpose(); // NOTE: result is a matrix (outer product)
}
void LinearLayer::UpdateDeltaBias(Eigen::RowVectorXd backward_input) {
        _DeltaBias += backward_input.transpose();
}

// Apply UpdateDeltaWeights to each column of forward input (which corresponds to data samples in a batch)
void LinearLayer::UpdateWeightsBias(double learning_rate=1.0) {
        if ((GetLayerCache().GetForwardInput() == nullptr) || (GetLayerCache().GetBackwardInput() == nullptr)) {
                throw std::invalid_argument("Pointer to forward/backward input is null!");
        }
        Eigen::MatrixXd& forward_input = *(GetLayerCache().GetForwardInput());
        Eigen::MatrixXd& backward_input = *(GetLayerCache().GetBackwardInput());
        // first set DeltaWeights and DeltaBias to zero
        ZeroDeltaBias();
        ZeroDeltaWeights();
        // loop over columns of forward input (each column corresponds to a data sample of the batch)
        for (int i=0; i<forward_input.cols(); i++) {
                UpdateDeltaWeights(forward_input.col(i), backward_input.row(i));
                UpdateDeltaBias(backward_input.row(i));
        }
        // add Deltas to the weights/bias
        // TODO: check sign!
        GetTransformation()->AddToWeights(learning_rate*_DeltaWeights);
        GetTransformation()->AddToBias(learning_rate*_DeltaBias);
}


