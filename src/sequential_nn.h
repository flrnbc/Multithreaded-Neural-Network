#ifndef SEQUENTIAL_NN_H_
#define SEQUENTIAL_NN_H_

#include <memory>
#include <vector>

class Layer;

class SequentialNN
/*
    Class encapsulating a sequential neural network.
*/
{
    private:
        // it turned out to be more convenient to work with a vector of shared
        // pointers (main reason: Layer has shared_ptr<Layer>'s as member var's)
        std::vector<std::shared_ptr<Layer> >  _pointers_layers;

    public: 
        SequentialNN(std::vector<Layer>); 
        // TODO #A: need reference? 
        std::vector<std::shared_ptr<Layer> > Layers() { return _pointers_layers; }     
        std::string Summary();
        // forward pass
        void Forward();
        // evaluate
        // TODO #B: overload ()-operator
        std::vector<double> Evaluate(std::vector<double>);

        // TODO #A: ADD flatten?
};

#endif // SEQUENTIAL_NN_H_