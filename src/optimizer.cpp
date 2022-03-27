#include "loss_function.h"
#include "optimizer.h"
#include "sequential_nn.h"
#include <Eigen/Dense>
#include <iostream>
#include <set>
#include <string>
#include <stdexcept>
#include <random>

// create random number
int SDG::createRandomNumber(int min, int max) {
    if (min > max) {
        throw std::invalid_argument("Not a valid interval for creating random numbers.");
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max); // for uniformly distributed ints in the closed interval [min, max]

    return dis(gen);
}

// create pairwise distinct random numbers by filling a set until it has distinctNumbers elements
// NOTE: this is not needed at the moment and will only be relevant for mini-batch training
std::set<int> SDG::createDistinctRandomNumbers(int min, int max, int distinctNumbers) {
    if (distinctNumbers > (max-min+1)) {
        throw std::invalid_argument("Too many distinct numbers.");
    }
    std::set<int> randomNums;
    while (randomNums.size() < distinctNumbers) {
        randomNums.insert(SDG::createRandomNumber(min, max)); // TODO: this might not be the most efficient way since sets are internally always kept sorted
    }
    return randomNums;
}

// single batch training step
// NOTE: this would be in the superclass if we created it at some point
void SDG::Step(SequentialNN& snn, const Eigen::MatrixXd& batch, const Eigen::MatrixXd& batchLabel) {
    // forward pass (also updates the LayerCaches etc.)
    snn(batch);
    // set input size and gradient of loss function if necessary
    // if (_lossFct->Cols() != snn.OutputSize()) {
    //     _lossFct->SetCols(snn.OutputSize());
    // }
    // loss
    double loss;
    loss = (*_lossFct)(snn.Output(), batchLabel);
    // TODO: output to file? At least give option for not printing all losses.
    //std::cout << "Loss: " << (*_lossFct)(snn.Output(), yLabel) << std::endl;

    //snn.Derivative(); // update derivatives in all layers
    // compute gradient of _lossFct with yLabel and the output of snn
    _lossFct->ZeroGrads();
    _lossFct->GradsAtPoints(snn.Output(), batchLabel);
    // update backward input of snn
    // NOTE: the minus sign is crucial!
    snn.BackwardInput(-_lossFct->GetGrads());
    // backward pass
    snn.Backward(); 
    // update weights/bias
    snn.UpdateWeightsBias(_learningRate);
    //snn.UpdateBias(_learningRate);
}


// trainining
void SDG::Train(SequentialNN& snn, const Eigen::MatrixXd& X, const Eigen::MatrixXd& yLabel, int epochs) {
    // number of data points/samples
    int totalSamples = X.cols();

    if (totalSamples != yLabel.cols()) {
        throw std::invalid_argument("Number of samples and labels do not coincide.");
    }
    if (totalSamples < _batchSize) {
        throw std::invalid_argument("Batch size exceeds number of samples.");
    }

    // data for batch training
    Eigen::MatrixXd batch = Eigen::MatrixXd::Zero(X.rows(), _batchSize);
    Eigen::MatrixXd batchLabel = Eigen::MatrixXd::Zero(yLabel.rows(), _batchSize);
 
    // loop over epochs
    for (int i=0; i<epochs; i++) {
        // create a random batch
        std::set<int> batchIndices = SDG::createDistinctRandomNumbers(0, totalSamples-1, _batchSize);        
        //index = SDG::createRandomNumber(0, N-1);
        int count=0;
        // TODO: range-loop was problematic
        for (std::set<int>::iterator it=batchIndices.begin(); it != batchIndices.end(); ++it) {
            batch.col(count) = X.col(*it);
            batchLabel.col(count) = yLabel.col(*it);
            count++;
        }
        // train with batch
        Step(snn, batch, batchLabel);
        // print loss
        if (i % 100 == 0) {
            std::cout << "=========== Epoch " << i+1 << " ===========" << std::endl;
            // TODO: add option to log to a file
            std::cout << "Loss: " << (*_lossFct)(snn.Output(), batchLabel) << std::endl;
            //std::cout << snn.Summary() << std::endl;
            //std::cout << "SNN backward output: " << snn.BackwardOutput() << std::endl; 
            //std::cout << "SNN output: \n" << snn.Output() << std::endl;
        }
    }
}