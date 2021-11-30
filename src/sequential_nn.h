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
        std::vector<std::shared_ptr<Layer> > _layers;

    public: 
        // constructor
        // TODO: need to change to std::vector<Layer> 
        SequentialNN(std::vector<std::shared_ptr<Layer> > layers); 
        
        // setters/getters
        // NOTE: return by reference to modify layers 'in place'.
        // TODO: check!
        //std::vector<Layer> GetLayers() { return _layers; }     
        int Length() { return _layers.size(); }

        // initialize the sequential network
        void Initialize();

        // forward pass
        // TODO #B: overload ()-operator
        void Input(std::vector<double>);
        void Forward();
        std::vector<double> Output();

        // summary
        std::string Summary();
        
        // evaluate
        //std::vector<double> Evaluate(std::vector<double>);

        // helper functions
        // check for composability of layers in a SequentialNN
        static bool ComposabilityCheck(const std::vector<std::shared_ptr<Layer> >&);
        // get initialization type of layers
        static std::string GetInitializationType(const std::shared_ptr<Layer>&, const std::shared_ptr<Layer>&);
};

#endif // SEQUENTIAL_NN_H_