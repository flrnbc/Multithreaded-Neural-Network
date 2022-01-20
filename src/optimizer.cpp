#include "loss_function.h"
#include "optimizer.h"
#include "sequential_nn.h"
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <stdexcept>
#include <random>

// single training step
void SDG::Step(SequentialNN& snn, const Eigen::VectorXd& X, const Eigen::VectorXd& yLabel) {
    // forward pass
    snn.Input(X);
    snn.Forward();

    // set input size of loss function
    _lossFct->SetCols(snn.OutputSize()); // TODO: good place?

    // loss
    // TODO: output to file?
    std::cout << "Loss: " << (*_lossFct)(snn.Output(), yLabel) << std::endl;

    // backward propagation 
    snn.UpdateDerivative(); // update derivatives in all layers
    // compute gradient of _lossFct with yLabel and the output of snn
    _lossFct->UpdateGradient(snn.Output(), yLabel);
    // update backward input of snn
    snn.BackwardInput(_lossFct->Gradient());
    // backward pass
    snn.Backward(); 
    // update weights/bias
    snn.UpdateWeights(_learningRate);
    snn.UpdateBias(_learningRate);
}


// trainining
void SDG::Train(SequentialNN& snn, const Eigen::MatrixXd& X, const Eigen::MatrixXd& yLabel, int epochs) {
    // number of data points/samples
    int N = X.cols();

    if (N != yLabel.cols()) {
        throw std::invalid_argument("Number of samples and labels do not coincide.");
    }
    // set input size (and hence size of gradient) of loss function
    _lossFct->SetCols(snn.OutputSize());

    // needed for stochastic gradient descent
    int randomCol;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, N-1); // for uniformly distributed ints in the closed interval [0, N-1]

    // loop over epochs
    for (int i=1; i<=epochs; i++) {
        std::cout << "=========== Epoch " << i << " ===========" << std::endl;
        randomCol = dis(gen);
        Step(snn, X.col(randomCol), yLabel.col(randomCol));
    }
}