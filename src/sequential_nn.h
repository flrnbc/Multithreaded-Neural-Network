#ifndef SEQUENTIAL_NN_H_
#define SEQUENTIAL_NN_H_

#include <memory>
#include "layer.h"
#include "loss_function.h"

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
        // NOTE: the constructor automatically initializes the weights (He/Xavier initialization)
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


        // helper functions
        // check for composability of layers in a SequentialNN
        static bool ComposabilityCheck(const std::vector<std::shared_ptr<Layer> >&);
        // get initialization type of layers
        static std::string GetInitializationType(const std::shared_ptr<Layer>&, const std::shared_ptr<Layer>&);
};

#endif // SEQUENTIAL_NN_H_