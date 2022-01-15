#include "optimizer.h"
#include "sequential_nn.h"
#include <Eigen/Dense>
#include <string>
#include <stdexcept>

// train method
SDG::Train(SequentialNN& snn, const Eigen::MatrixXd& X, const Eigen::MatrixXd& yLabel) {
    // number of data points/samples
    int N = X.cols();

    if (N != yLabel.cols()) {
        throw std::invalid_argument("Number of samples and labels do not coincide.")
    }
    // set input size (and hence size of gradient) of loss function
    _lossFct->SetCols(snn.OutputSize());

    // needed for stochastic gradient descent
    int randomCol;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, N-1); // for uniformly distributed ints in the closed interval [0, N-1]

    // loop over epochs
    for (int i=1; i<=NumberEpochs(); i++) {
        std::cout << "=========== Epoch " << i << " ===========" << std::endl;
        randomCol = dis(gen);
        snn.Train(*(_lossFct), LearningRate(), X.col(randomCol), yLabel.col(randomCol));
    }
}