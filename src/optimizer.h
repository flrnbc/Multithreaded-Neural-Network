#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <Eigen/Dense>
#include "loss_function.h"
#include "sequential_nn.h"
#include <string>
#include <stdexcept>

// NOTE: for now only one optimizer, namely stochastic gradient descent.
// We might add a class hierarchy later if we introduce more optimizers.

class SDG {
    private:
        // loss function for optimzation
        std::unique_ptr<LossFunction> _lossFct; 
        double _learningRate;

    public:
        // constructor
        SDG(std::string lossFctName, double learningRate): 
            _lossFct(std::make_unique<LossFunction>(LossFunction(lossFctName))),
            _learningRate(learningRate) {}

        // setters
        void SetLearningRate(double a) { _learningRate = a; }
        
        // getters
        double LearningRate() { return _learningRate; }

        // optimize
        // single training step
        void Step(SequentialNN&, const Eigen::VectorXd&, const Eigen::VectorXd&);

        // here training is just online learning (i.e. mini-batch learning with batch size = 1)
        void Train(SequentialNN&, const Eigen::MatrixXd&, const Eigen::MatrixXd&, int epochs);
};

#endif // OPTIMIZER_H_