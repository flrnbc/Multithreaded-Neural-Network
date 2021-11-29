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
        std::shared_ptr<std::vector<double> > _forward_input;
        std::shared_ptr<std::vector<double> > _forward_output;
        std::shared_ptr<std::vector<double> > _backward_input;
        std::shared_ptr<std::vector<double> > _backward_output;

     public:
        // constructor
        LayerCache() {}
        // destructor
        ~LayerCache() {}

        // TODO: again the default copy/move constructors/assignment operators should be enough?!?
        // (i.e. use the ones for std::vector)

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
        void ConnectForward(int, LayerCache&); 
        // backward
        // void ConnectBackward(Layer);

       
};

#endif // LAYER_CACHE_H_