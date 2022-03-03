#ifndef LAYER_CACHE_H_
#define LAYER_CACHE_H_

#include <Eigen/Dense>
#include <memory>

/****************
 * LAYER CACHE *
 ****************/

/**
    The LayerCache class is the storage of a layer for the forward/backward input and output used in forward and 
    backward pass. Of course, we could save these in the Layer class themselves. However, this is inefficient since e.g. the 
    forward output of one layer is the forward input of the next layer. The LayerCache only points to the corresponding data 
    and we can connect two LayerCaches together by pointing to the same data. That's why we use shared_ptrs. Moreover, we 
    separate the data (handled in LayerCache) from the transformations in a layer.

    Note: it might be sufficient to work with references for sequential neural networks. But LayerCache is more flexible. 
    For example, we could use it for other network topologies (by adjusting the Connect methods correspondingly).

    TODO: might be useful to separate forward and backward cache.
*/ 

class LayerCache {
    private:
        /** Cache used in forward pass:
            Stores the in-/output of a layer. Here each data point is a _column vector_.
            In particular, the number of columns of the Eigen::MatrixXd coincides with the number of 
            data samples in a batch.
        */
        std::shared_ptr<Eigen::MatrixXd> _forward_input;
        std::shared_ptr<Eigen::MatrixXd> _forward_output;

        /** Cache for backward pass:
            keeps track of the loss function gradients which are propagated backwards in a neural
            network. NOTE: we consider gradients as row vectors. In particuar, the i-th row corresponds
            to the loss function gradient for the i-th data sample in a batch.
        */
        std::shared_ptr<Eigen::MatrixXd> _backward_input;
        std::shared_ptr<Eigen::MatrixXd> _backward_output;

     public:
        // constructor
        LayerCache():
            _forward_input(nullptr), 
            _forward_output(nullptr), 
            _backward_input(nullptr),
            _backward_output(nullptr) {}

        // destructor
        ~LayerCache() {}

        // copy and move semantics
        // TODO: don't the default ones from Eigen suffice?

        // copy constructor
        LayerCache(const LayerCache& source) {
            _forward_input = source._forward_input;
            _forward_output = source._forward_output;
            _backward_input = source._backward_input;
            _backward_output = source._backward_output;
        }
        
        // copy assignment operator
        LayerCache &operator=(const LayerCache& source) {
            if (this == &source) {
                return *this;
            }
            // TODO: need to reset the shared_ptrs first?
            _forward_input = source._forward_input;
            _forward_output = source._forward_output;
            _backward_input = source._backward_input;
            _backward_output = source._backward_output;

            return *this;
        }
        
        // move constructor
        LayerCache(LayerCache &&source) = default;

        // move assignment operator
        LayerCache &operator=(LayerCache &&source) = default;

        // setters/getters 
        // forward
        void SetForwardInput(std::shared_ptr<Eigen::MatrixXd> input_ptr); 
        void SetForwardOutput(std::shared_ptr<Eigen::MatrixXd> output_ptr);
        std::shared_ptr<Eigen::MatrixXd > GetForwardOutput();
        std::shared_ptr<Eigen::MatrixXd > GetForwardInput();

        // backward
        void SetBackwardInput(std::shared_ptr<Eigen::MatrixXd> backward_input_ptr);
        void SetBackwardOutput(std::shared_ptr<Eigen::MatrixXd> output_ptr);
        std::shared_ptr<Eigen::MatrixXd> GetBackwardOutput();
        std::shared_ptr<Eigen::MatrixXd> GetBackwardInput();

        // set (mini-)batch size (number of columns in the forward cache or rows in the backward cache)
        // NOTE: sets values to zero if the batch size is actually changed 
        void SetBatchSize(int);

        /** Connecting this LayerCache with another one from 'left to right', i.e. 
            the forward output of *this will be connected to the forward input of 
            next_layer_cache etc. This is mainly adjusted to sequential neural 
            networks at the moment.

            NOTE: Also instantiates a zero vector of the given size at which the 
            pointers point.
        */
        // forward
        void ConnectForward(int size_forward_output, int number_samples, LayerCache& next_layer_cache); 
        // backward
        void ConnectBackward(int size_backward_output, int number_samples, LayerCache& next_layer_cache);
        // both 
        void Connect(int size_forward_output, int number_samples, int size_backward_output, LayerCache&);
};

#endif // LAYER_CACHE_H_