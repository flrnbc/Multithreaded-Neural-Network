#ifndef SEQUENTIAL_NN_H_
#define SEQUENTIAL_NN_H_

#include <memory>
#include "layer.h"
#include <vector>

class Layer;

class SequentialNN
/*
    Class encapsulating a sequential neural network.
*/
{
    private:
        std::vector<std::shared_ptr<Layer> >  _layers;

    public: 
        // constructor
        SequentialNN(std::vector<std::shared_ptr<Layer> >); 
        
        // setters/getters
        // NOTE: return by reference to modify layers 'in place'.
        // TODO: check!
        //std::vector<Layer> GetLayers() { return _layers; }     
        //std::string Summary();
        int Length() { return _layers.size(); }

        // forward pass
        // TODO #B: overload ()-operator
        void Input(std::vector<double>);
        void Forward();
        
        // evaluate
        //std::vector<double> Evaluate(std::vector<double>);
};

#endif // SEQUENTIAL_NN_H_