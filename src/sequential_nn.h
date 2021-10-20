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
        std::string Summary(); 

};



#endif // SEQUENTIAL_NN_H_