#ifndef LAYER_CACHE_H_
#define LAYER_CACHE_H_

#include <Eigen/Dense>
#include <memory>

/****************
 * LAYER CACHE *
 ****************/

/*
    The LayerCache class is the storage of a layer for the forward/backward input and output. Of course, we could save
    these in the Layer class themselves. However, this is inefficient since e.g. the forward output of one layer is the
    forward input of the next layer. The LayerCache only points to the corresponding data and we can connect two LayerCaches
    together by pointing to the same data. That's why we use shared_ptrs.

    Note: it might be sufficient to work with references for sequential neural networks. But LayerCache is more flexible. 
    For example, we could use it for other network topologies.

    Improvement: use templates (e.g. for col and row vectors) so that we can separate the forward and backward cache. 
*/ 

class LayerCache {
    private:
        // cache for forward pass
        // NOTE: each data point is a column vector here
        std::shared_ptr<Eigen::VectorXd> _forward_input;
        std::shared_ptr<Eigen::VectorXd> _forward_output;

        // cache for backward pass, i.e. keeps track of the gradients (\Delta_i in documentation)
        // NOTE: we consider gradients as row vectors
        std::shared_ptr<Eigen::RowVectorXd> _backward_input;
        std::shared_ptr<Eigen::RowVectorXd> _backward_output;

     public:
        // constructor (nullptr added to be explicit)
        LayerCache(): 
            _forward_input(nullptr), 
            _forward_output(nullptr), 
            _backward_input(nullptr),
            _backward_output(nullptr) {}

        // destructor
        ~LayerCache() {}

        // copy and move semantics
        // TODO: don't the default ones suffice?

        //copy constructor
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
        void SetForwardInput(std::shared_ptr<Eigen::VectorXd> input_ptr); 
        void SetForwardOutput(std::shared_ptr<Eigen::VectorXd> output_ptr);
        std::shared_ptr<Eigen::VectorXd > GetForwardOutput();
        std::shared_ptr<Eigen::VectorXd > GetForwardInput();

        // backward
        void SetBackwardInput(std::shared_ptr<Eigen::RowVectorXd> backward_input_ptr);
        void SetBackwardOutput(std::shared_ptr<Eigen::RowVectorXd> output_ptr);
        std::shared_ptr<Eigen::RowVectorXd> GetBackwardOutput();
        std::shared_ptr<Eigen::RowVectorXd> GetBackwardInput();

        // connecting Layer's 
        // forward
        void ConnectForward(int, LayerCache&); 
        // backward
        void ConnectBackward(int, LayerCache&);
        // both 
        void Connect(int, int, LayerCache&);
};

#endif // LAYER_CACHE_H_