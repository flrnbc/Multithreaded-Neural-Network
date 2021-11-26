#ifndef SEQUENTIAL_NN_H_
#define SEQUENTIAL_NN_H_

#include <memory>
#include "layer.h"
#include <vector>


class SequentialNN
/*
    Class encapsulating a sequential neural network.
*/
{
    private:
        std::vector<Layer>  _layers;

    public: 
        // constructor
        SequentialNN(std::vector<Layer>); 
        
        // setters/getters
        // NOTE: we do not allow changing the Layer's on purpose
        // TODO #A: need reference? 
        std::vector<Layer> Layers() { return _layers; }     
        std::string Summary();
        
        

        // forward pass
        // TODO #B: overload ()-operator
        void SetInput(std::vector<double>);
        void Forward();
        
        // evaluate
        //std::vector<double> Evaluate(std::vector<double>);
};

#endif // SEQUENTIAL_NN_H_