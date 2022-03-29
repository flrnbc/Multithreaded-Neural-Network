#include "layer_cache.h"
#include <memory>
#include <vector>

/**************
 * LAYER CACHE *
 **************/

// TODO: would it be better to give other options to set the corresponding vectors?

/** Change batch size and reset to the zero matrix of the corresponding size.
    If rows=true, resize the rows to number_samples.
    If rows=false, resize the cols to number_samples.
    In case number_samples already coincides with the already set numbers, don't do anything.
*/
void ResizeAndZero(int number_samples, std::shared_ptr<Eigen::MatrixXd> matrix_ptr, bool rows) {
    if (matrix_ptr == nullptr) {
        // We do not throw an error here because we can ignore this case in our application to sequential NNs.
        return; 
    }
    if ((rows) && (matrix_ptr->rows() != number_samples)) { 
        matrix_ptr->setZero(number_samples, matrix_ptr->cols()); 
    } 
    else if (!(rows) && (matrix_ptr->cols() != number_samples)) {
        matrix_ptr->setZero(matrix_ptr->rows(), number_samples);
    }
}

void LayerCache::SetBatchSize(int number_samples) {
    // batch size = number of cols in the forward pass; hence pass rows=false to ResizeAndZero
    ResizeAndZero(number_samples, _forward_input, false);
    ResizeAndZero(number_samples, _forward_output, false);
    // batch size = number of rows in the backward pass; so row=true
    ResizeAndZero(number_samples, _backward_input, true);
    ResizeAndZero(number_samples, _backward_output, true);

}

// connecting layer caches ('from left to right')
// forward
void LayerCache::ConnectForward(int size_forward_output, int number_samples, LayerCache& next_layer_cache) {
    Eigen::MatrixXd zero_matrix = Eigen::MatrixXd::Zero(size_forward_output, number_samples);

    // TODO: might need a check if SetForwardOutput is not null, e.g. if we add 
    // the option to add/delete layers
    this->SetForwardOutput(std::make_shared<Eigen::MatrixXd>(zero_matrix));
    next_layer_cache.SetForwardInput(this->GetForwardOutput());
}

// backward
// NOTE: since we propagate backwards, the backward input of *this points to the same
// vector as the backward output of next_layer_cache.
void LayerCache::ConnectBackward(int size_backward_input, int number_samples, LayerCache& next_layer_cache) {
    Eigen::MatrixXd zero_matrix = Eigen::MatrixXd::Zero(number_samples, size_backward_input);

    this->SetBackwardInput(std::make_shared<Eigen::MatrixXd>(zero_matrix));
    next_layer_cache.SetBackwardOutput(this->GetBackwardInput());
}

// connect both
void LayerCache::Connect(int size_forward_output, int size_backward_input, int number_samples, LayerCache& next_layer_cache) {
    ConnectForward(size_forward_output, number_samples, next_layer_cache);
    ConnectBackward(size_backward_input, number_samples, next_layer_cache);
}

