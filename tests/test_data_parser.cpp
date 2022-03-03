#include <Eigen/Dense>
#include "../src/data_parser.h"
#include <iostream>
#include <fstream>
#include <vector>

void test_data_parser() {
    DataParser dp;

    Eigen::MatrixXd matrix = dp.LoadCSV<Eigen::MatrixXd>("./tests/train.csv"); // path relative to tests directory
    std::cout << matrix << std::endl;
    matrix = matrix(Eigen::placeholders::all, Eigen::seq(1, Eigen::indexing::last));
    std::cout << matrix.row(0) << std::endl;

    Eigen::VectorXd w{{1, 2, 3, 3, 3, 1, 2, 3, 2, 1, 1, 1, 0, 0, 0}};
    // cast
    std::cout << dp.OneHotEncoder(w, 3) << std::endl;
}

void test_MNIST() {
    DataParser dp;

    Eigen::MatrixXd X = dp.LoadCSV<Eigen::MatrixXd>("./tests/MNIST_train.csv");
    std::cout << "Cols of X: " << X.cols() << std::endl;

    Eigen::MatrixXd yLabel;
    yLabel = dp.OneHotEncoder(X.col(0), 9);
    std::cout << yLabel.rows() << " " << yLabel.cols() << std::endl;
    std::cout << yLabel.row(0) << std::endl;

    // 'remove' first columns (the labels)
    // NOTE: this is not odeal because we create a copy... However, X = X(Eigen...) gave a segmentation fault which I couldn't resolve.
    Eigen::MatrixXd X_train = X(Eigen::placeholders::all, Eigen::seq(1, Eigen::indexing::last));
    std::cout << "Cols of X_train: " << X_train.cols() << std::endl;
}

int main() {
    test_data_parser();
    //test_MNIST();

    return 0;
}