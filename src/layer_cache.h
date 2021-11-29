#ifndef LAYER_CACHE_H_
#define LAYER_CACHE_H_

#include <memory>
#include <vector>

/****************
 * LAYER CACHE *
 ****************/

class LayerCache {
    private:
        // ownership important: the corresponding resources will be shared with different pointers
        // (namely in layers of sequential neural nets)
        // TODO: might be better to split in forward and backward cache
        std::shared_ptr<std::vector<double> > _forward_input;
        std::shared_ptr<std::vector<double> > _forward_output;
        std::shared_ptr<std::vector<double> > _backward_input;
        std::shared_ptr<std::vector<double> > _backward_output;

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
        // NOTE: backward/forward output should be computed, NOT set.
        // forward
        void SetForwardInput(std::shared_ptr<std::vector<double> > input_ptr); 
        void SetForwardOutput(std::shared_ptr<std::vector<double> > output_ptr);
        std::shared_ptr<std::vector<double> > GetForwardOutput();
        std::shared_ptr<std::vector<double> > GetForwardInput();
        // backward
        void SetBackwardInput(std::shared_ptr<std::vector<double> > backward_input_ptr);
        std::shared_ptr<std::vector<double> > GetBackwardOutput();

        // connecting Layer's 
        // forward
        void ConnectForward(int, std::shared_ptr<LayerCache>&); 
        // backward
        // void ConnectBackward(Layer);

       
};

#endif // LAYER_CACHE_H_