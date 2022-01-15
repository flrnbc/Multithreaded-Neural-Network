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
        std::unique_ptr<LossFunction> _lossFct;
        double _learningRate;
        int _numberEpochs;

    public:
        // constructor
        SDG(std::string lossFctName, double learningRate, int numberEpochs): 
            _lossFct(std::make_unique<LossFunction>(LossFunction(lossFctName))),
            _learningRate(learningRate),
            _numberEpochs(numberEpochs)
            {
                if (numberEpochs < 0) _numberEpochs = 0; // TODO: better way to deal with negative input?
            }

        // setters
        void SetLearningRate(double a) {
            if (a <= 0) {
                throw std::invalid_argument("Learning rate is not positive.");
            } 
            _learningRate = a; 
        }
        void SetNumberEpochs(int N) { 
            if (N < 1) {
                throw std::invalid_argument("Number of epoch has to be at least 1.");
            }
            _numberEpochs = N; 
        }

        // getters
        double LearningRate() { return _learningRate; }
        double NumberEpochs() { return _numberEpochs; }

        // optimize
        void Train(SequentialNN&, const Eigen::MatrixXd&, const Eigen::MatrixXd&);
};

#endif // OPTIMIZER_H_