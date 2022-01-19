#ifndef SEQUENTIAL_NN_H_
#define SEQUENTIAL_NN_H_

#include <memory>
#include "layer.h"
#include "loss_function.h"
#include <stdexcept>

/*****************************
 * SEQUENTIAL NEURAL NETWORK *
 *****************************/

/**
    Class to encapsulate a sequential neural network. Here 'sequential' means that we have a stack
    of Layers, i.e. all hidden Layers have precisely one input and output Layer. 

    TODO: Change initialization!
 */


// forward declaration
class Layer;

class SequentialNN {
    private:
        // shared_ptrs to Layers of sequential NN
        std::vector<std::shared_ptr<Layer> > _layers; // TODO: use unique_ptrs instead?
        
        // helper functions //
        // check for composability of layers in a SequentialNN
        static bool ComposabilityCheck(const std::vector<std::shared_ptr<Layer> >&);
        // convert vector of Layers to pass to second constructor
        static std::vector<std::shared_ptr<Layer> > ConvertToSharedPtrs(const std::vector<Layer>& vlayers) {
            std::vector<std::shared_ptr<Layer>> ptrs_layers;
            for (int i=0; i<vlayers.size(); i++) {
                ptrs_layers.emplace_back(std::make_shared<Layer>(vlayers[i]));
            }
            return ptrs_layers;
        }

    public: 
        // constructor with vector of shared_ptrs to Layers
        // NOTE: the constructor automatically initializes the weights (He/Xavier initialization)
        SequentialNN(std::vector<std::shared_ptr<Layer> >); 

        // constructor with vector of Layers (using delegating constructor (introduced in C++11))
        SequentialNN(std::vector<Layer> vlayers): SequentialNN(ConvertToSharedPtrs(vlayers)) {}

        // setters/getters   
        int Length() { return _layers.size(); }
        // get input and output size
        int InputSize();
        int OutputSize();

        // initialize the sequential network
        void Initialize();

        // forward pass
        // TODO #B: overload ()-operator
        // TODO: better use (const) reference/move semantics?
        void Input(Eigen::VectorXd);
        void Forward();
        Eigen::VectorXd& Output();

        // backward pass
        void UpdateDerivative(); // update derivative with forward LayerCache
        void BackwardInput(Eigen::RowVectorXd);
        void Backward();
        Eigen::RowVectorXd BackwardOutput();

        // compute loss
        double Loss(LossFunction&, const Eigen::VectorXd& yLabel);
        void UpdateBackwardInput(LossFunction&, const Eigen::VectorXd& yLabel);

        // update weights/bias
        void UpdateWeights(double);
        void UpdateBias(double);

        // single train cycle
        void Train(LossFunction& lossFct, double learning_rate, const Eigen::VectorXd& input, const Eigen::VectorXd& yLabel);

        // summary
        std::string Summary();

        // helper function
        // get initialization type of layers
        static std::string GetInitializationType(const std::shared_ptr<Layer>&, const std::shared_ptr<Layer>&);
};

#endif // SEQUENTIAL_NN_H_