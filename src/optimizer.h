#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <Eigen/Dense>
#include "loss_function.h"
#include "sequential_nn.h"
#include <set>
#include <string>
#include <stdexcept>

/** 
    Optimizer class (currently only stochastic gradient descent) for training 
    sequential neural networks using backpropagation. If ~snn~ is an object of
    SequentialNN, then it is trained using the mean squared error as loss function
    and training data ~X~ (Eigen::MatrixXd) with labels ~yLabel~ (Eigen::MatrixXd)
    as follows:
        double learningRate = 0.0001;
        int batchSize = 5;
        int numberOfEpochs = 5000;
        SDG sdg("mse", batchSize, learningRate);
        // train the model
        sdg.Train(snn, X, y, numberOfEpochs);

    There is also the option for a single training step using ~Step~.

    NOTE: for now we only offer one optimizer, namely stochastic gradient descent.
          We might add a class hierarchy later if we introduce more optimizers.


    TODO: it might be better to include (references to) snn, X, yLabel into the optimizer class to avoid
          methods with a very long list of parameters
*/ 

class SDG {
    private:
        // loss function for optimzation
        std::unique_ptr<LossFunction> _lossFct; 
        double _learningRate;
        // batch size
        int _batchSize;
        // create random number
        static int createRandomNumber(int min, int max);
        // create pairwise distinct numbers
        static std::set<int> createDistinctRandomNumbers(int min, int max, int distinctNumbers);

    public:
        // constructor
        SDG(std::string lossFctName, int batchSize, double learningRate): 
            _lossFct(std::make_unique<LossFunction>(LossFunction(lossFctName))),
            _learningRate(learningRate), 
            _batchSize(batchSize) {}

        // setters
        void SetLearningRate(double a) { _learningRate=a; }
        void SetBatchSize(int);
        
        // getters
        double LearningRate() { return _learningRate; }

        // optimize
        // single step of stochastic gradient descent (mini-batch learning)
        void Step(SequentialNN&, const Eigen::MatrixXd&, const Eigen::MatrixXd&);

        // online-training
        void Train(SequentialNN&, const Eigen::MatrixXd&, const Eigen::MatrixXd&, int epochs);
};

#endif // OPTIMIZER_H_