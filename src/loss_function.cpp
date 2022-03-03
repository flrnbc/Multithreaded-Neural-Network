#include "loss_function.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

// constructor
std::vector<std::string> loss_functions = {"mse", "cross_entropy"};

LossFunction::LossFunction(std::string name) {
    auto find_loss_fct = std::find(std::begin(loss_functions), std::end(loss_functions), name);
    if (find_loss_fct == loss_functions.end()) {
        throw std::invalid_argument("Unknown loss function.");
    }
    _name = name;
    _gradients = Eigen::MatrixXd::Zero(0, 0); // empty gradients
}

// check size of input vectors and compare with ~cols~ (input size)
void LossFunction::CheckSize(Eigen::MatrixXd& y, const Eigen::MatrixXd& yLabel) {
     if ((y.cols() != yLabel.cols()) || (y.rows() != yLabel.rows())) {
        throw std::invalid_argument("Shapes of predictions and labels do not coincide.");
    }
}

double Log(double x) {
    return std::log(x);
}

// TODO: the following methods need to be refactored at some point (e.g. by making LossFunction abstract and MSE etc. concrete)
// overload () by evaluating the prediction and corresponding label
double LossFunction::operator()(Eigen::MatrixXd& y, const Eigen::MatrixXd& yLabel) {
    CheckSize(y, yLabel);
    double loss = 0;
     
    // any of the following cases is guaranteed by the constructor
    if (Name() == "mse") {
        double avg = 1/double(y.rows());
        for (int i=0; i<y.cols(); i++) {
            loss += ((y-yLabel).col(i)).dot((y-yLabel).col(i)); // scalar/dot product
        }
        return avg*loss;
    }
    else if (Name() == "cross_entropy") {
        Eigen::MatrixXd yLog = Eigen::MatrixXd::Zero(y.rows(), y.cols()); 
        // TODO: did not mange to get Eigen's unaryExpr to work...
        for (int j=0; j<y.cols(); j++) {
            for (int i=0; i<y.rows(); i++) {
                yLog(i, j) = Log(y(i, j)+1e-9); // off-set to avoid log(0)
            }
            loss += (-yLog.col(j)).dot(yLabel.col(j)); 
        }
        return loss;
    }
}

// update gradients of loss function at given vectors
void LossFunction::GradsAtPoints(Eigen::MatrixXd& y, const Eigen::MatrixXd& yLabel) {
    CheckSize(y, yLabel);
    // reset gradients to zero matrix of correct size if necessary
    if ((_gradients.cols() != y.rows()) || (_gradients.rows() != y.cols())) {
        _gradients = Eigen::MatrixXd::Zero(y.cols(), y.rows());
    }
    // any of the following cases is guaranteed by the constructor
    if (Name() == "mse") { 
        // i-th row is the gradient with respect to the i-th column of y (no derivative in yLabel-direction)
        _gradients = (2/double(y.rows()))*(y-yLabel).transpose();
    }
    else if (Name() == "cross_entropy") {
        Eigen::MatrixXd yDerived = Eigen::MatrixXd::Zero(y.cols(), y.rows());
        for (int j=0; j<y.cols(); j++) {
            for (int i=0; i<y.rows(); i++) {
                // TODO: check formula!
                yDerived(j, i) = -yLabel(i, j)/(y(i, j)+1e-9); // off-set to avoid division by zero (y(i)>=0 because it is output of softmax)
            }
        }
        _gradients = yDerived;
    }
}

// set gradients to zero
void LossFunction::ZeroGrads() {
    int rows = _gradients.rows();
    int cols = _gradients.cols();
    _gradients = Eigen::MatrixXd::Zero(rows, cols);
}
