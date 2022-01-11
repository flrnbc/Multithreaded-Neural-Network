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
        std::unique<LossFunction> _lossFct;
        double _learningRate;
        int _numberEpochs;

    public:
        // constructor
        SDG(std::string lossFctName, double learningRate, int numberEpochs): 
            _lossFct(std::make_unique<LossFunction>(LossFunction(lossFctName))),
            SetLearningRate(learningRate),
            SetNumberEpochs(numberEpochs)
            {}

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
        Train(SequentialNN&, const Eigen::MatrixXd&, const Eigen::MatrixXd&);
};

#endif // OPTIMIZER_H_