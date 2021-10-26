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
        std::unique_ptr<std::vector<Layer> > _layers_ptr;

    public: 
        SequentialNN(std::vector<Layer>);  
        std::unique_ptr<std::vector<Layer> >& Layers() { return _layers_ptr; }     
        std::string Summary();
        // forward pass
        void Forward();
        // evaluate
        // TODO #B: overload ()-operator
        std::vector<double> Evaluate(std::vector<double>);

        // TODO #A: ADD flatten?
};

#endif // SEQUENTIAL_NN_H_