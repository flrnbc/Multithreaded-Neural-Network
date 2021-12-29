#ifndef LAYER_CACHE_H_
#define LAYER_CACHE_H_

#include <Eigen/Dense>
#include <memory>

/****************
 * LAYER CACHE *
 ****************/

class LayerCache {
    private:
        // ownership important: the corresponding resources will be shared with different pointers
        // (namely in layers of sequential neural nets)
        // TODO: might be better to split in forward and backward cache

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
        void SetBackwardOutput(std::shared_ptr<Eigen::VectorXd> output_ptr);
        std::shared_ptr<Eigen::RowVectorXd> GetBackwardOutput();
        std::shared_ptr<Eigen::RowVectorXd> GetBackwardInput();

        // connecting Layer's 
        // forward
        void ConnectForward(int, LayerCache&); 
        // backward
        void ConnectBackward(int, LayerCache&);
};

#endif // LAYER_CACHE_H_