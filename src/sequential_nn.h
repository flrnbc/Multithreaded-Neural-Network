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
    Class to encapsulate a sequential neural network (SNN). Here 'sequential' means that we have a stack
    of Layers, i.e. all hidden Layers have precisely one input and output Layer. 

    To initialize an SNN, we first initialize a vector of Layers, e.g. 
        std::vector<Layer> vlayers{{LinearLayer(8, 16), 
                                    ActivationLayer(8, "relu"),
                                    LinearLayer(4, 8),
                                    ActivationLayer(4, "softmax")
                                    }};
    Then ~vlayer~ is passed to the constructor
        SequentialNN snn(vlayers);
    This already initializes the weights of the LinearLayers according to the corresponding consequent ActivationLayers
    (He or Xavier initialization).

    Next we summarize the most important member functions:

    Retrieving information of ~snn~:
        * ~snn.Length()~: returns the number of Layers
        * ~snn.InputSize()~: input size of the first Layer
        * ~snn.OutputSize()~: output size of the last layer
        * ~snn.Summary()~: summary of snn (summaries of each layer which is very verbose at the moment)
    
    Evaluating a vector (~Eigen::VectorXd~) ~input~ of the correct size:
        ~snn(input)~

    The remaining methods are for the forward and backward pass.

    Forward pass:
        * ~Input~: sets the forward input of the first layer
        * ~Forward~: forward pass through all layers
        * ~Output~: returns reference to forward output of last layer
    Backward pass: 
        * ~BackwardInput~: sets the backward input of the last layer
        * ~Backward~: backward pass through all layers using the derivatives of the layers
        * ~BackwardOutput~: returns reference to backward output of first layer
    
    Finally, we can update the weights and bias with ~UpdateWeightsBias~ according to the backpropagation algorithm.
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
        // NOTE: the constructor automatically initializes the weights (He/Xavier initialization) using ~Initialize()~
        SequentialNN(std::vector<std::shared_ptr<Layer> >); 

        // constructor with vector of Layers (using delegating constructor (introduced in C++11))
        SequentialNN(std::vector<Layer> vlayers): SequentialNN(ConvertToSharedPtrs(vlayers)) {}

        // setters/getters   
        int Length() { return _layers.size(); }
        // get input and output size
        int InputSize();
        int OutputSize();
        // summary
        std::string Summary();

        // TODO: some of the following member functions need to be made private (e.g. Forward(), Output()) because
        // they are not intended to be used by the user

        // initialize the sequential network (called by constructors)
        void Initialize();

        // forward pass
        // TODO: better use (const) reference/move semantics?
        void Input(Eigen::MatrixXd);
        void Forward();
        Eigen::MatrixXd& Output(); // TODO: check if reference is really useful/reasonable here
        Eigen::MatrixXd operator()(Eigen::MatrixXd);

        // backward pass
        //void Derivative(); // update derivative with forward LayerCache
        void BackwardInput(Eigen::MatrixXd);
        void Backward();
        Eigen::MatrixXd BackwardOutput();

        // update weights/bias
        void UpdateWeightsBias(double learning_rate);

        // helper function
        // get initialization type of layers
        static std::string GetInitializationType(const std::shared_ptr<Layer>&, const std::shared_ptr<Layer>&);
};

#endif // SEQUENTIAL_NN_H_