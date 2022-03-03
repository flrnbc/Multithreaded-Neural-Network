#include "src/data_parser.h"
#include "src/layer.h"
#include "src/loss_function.h"
#include "src/sequential_nn.h"
#include "src/optimizer.h"
#include <Eigen/Dense>
#include <iostream>
#include <utility>
#include <vector>

/** Application of the classes ~SequentialNN~, ~Optimizer~ and ~DataParser~ to train a neural network
    for the MNIST dataset (handwritten integers from 0 to 9).

*/

// split data samples and labels (for now each label is a single number; hence the labels form a row vector)
std::pair<Eigen::MatrixXd, Eigen::RowVectorXd> split_samples_labels(const std::string file_path) {
    DataParser dp;
    Eigen::MatrixXd X = dp.LoadCSV<Eigen::MatrixXd>(file_path);

    // get the actual training samples
    // we work with column vectors as data 
    // TODO: needs to be changed at some point because most data is provided as row vectors)
    Eigen::MatrixXd samples = X(Eigen::placeholders::all, Eigen::seq(1, Eigen::indexing::last)).transpose(); 
    // get labels
    Eigen::RowVectorXd labels = X.col(0).transpose();

    return {samples, labels};
}

// train on data
void train_MNIST(SequentialNN& snn, int batchSize, int epochs, std::string file_path) {
    DataParser dp;
    std::pair<Eigen::MatrixXd, Eigen::RowVectorXd> train_data = split_samples_labels(file_path);
    auto sdg = SDG("cross_entropy", batchSize, 0.003);
    // one-hot-encode before training
    sdg.Train(snn, train_data.first, dp.OneHotEncoder(train_data.second, 9), epochs);
}

// evaluate against test data
void evaluate_test(SequentialNN& snn, Eigen::MatrixXd X_test, Eigen::RowVectorXd y_test) {
    int total = X_test.cols();
    int correct = 0;
    Eigen::MatrixXd::Index max_index;

    for (int i=0; i<total; i++) { // TODO: how to improve?
        if ((snn(X_test.col(i)).col(0)).maxCoeff(&max_index) == (int)y_test(i)) { // CHECK!
            correct++;
        }
    }
    std::cout << "Correct: " << correct << "/" << total << std::endl;
}

void test_MNIST(SequentialNN& snn, std::string file_path) {
    std::pair<Eigen::MatrixXd, Eigen::RowVectorXd> test_data = split_samples_labels(file_path);
    evaluate_test(snn, test_data.first, test_data.second);
}

// training and evaluating a neural network on the MNIST data set
int main() {
    // setup a basic neural network
    std::vector<Layer> layers{{LinearLayer(392, 784), 
                               ActivationLayer(392, "relu"), 
                               LinearLayer(196, 392), 
                               ActivationLayer(196, "relu"),
                               LinearLayer(10, 196), 
                               ActivationLayer(10, "softmax")}};
    auto snn = SequentialNN(layers);

    // training
    train_MNIST(snn, 50, 200, "./tests/MNIST_train.csv");
    // evaluate
    test_MNIST(snn, "./tests/MNIST_test.csv");

    return 0;
}