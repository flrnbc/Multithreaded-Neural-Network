#include "loss_function.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// setter(s)
// NOTE: we always ensure that the size of the gradient matches the input size (cols)
// of the loss function
void LossFunction::SetCols(int cols) {
    _cols = cols;

    // initialize gradient or reset if necessary
    if (_gradient == nullptr) {
        _gradient = std::make_unique<Eigen::RowVectorXd>(Eigen::RowVectorXd::Zero(cols));
    } 
    else if (_gradient->size() != cols) {
        *_gradient = Eigen::RowVectorXd::Zero(cols);
    }
}


// constructor
std::vector<std::string> loss_functions = {"mse"};

LossFunction::LossFunction(std::string name) {
    auto find_loss_fct = std::find(std::begin(loss_functions), std::end(loss_functions), name);
    if (find_loss_fct == loss_functions.end()) {
        throw std::invalid_argument("Unknown loss function.");
    }
    _name = name;
}

// methods
void LossFunction::CheckSize(Eigen::VectorXd& y, const Eigen::VectorXd& yLabel) {
     if (y.size() != yLabel.size()) {
        throw std::invalid_argument("Vectors are not of same size.");
    }
    if (y.size() != Cols()) {
        throw std::invalid_argument("Size of vectors does not coincide with the input size of the loss function.");
    }
}



double LossFunction::operator()(Eigen::VectorXd& y, const Eigen::VectorXd& yLabel) {
    CheckSize(y, yLabel);

    // any of the following cases is guaranteed by the constructor
    if (Name() == "mse") {
        double avg = 1/double(y.size());
        double scalar_prod = ((y-yLabel).transpose()*(y-yLabel))(0);

        return avg*scalar_prod;
    }
}

void LossFunction::UpdateGradient(Eigen::VectorXd& y, const Eigen::VectorXd& yLabel) {
    CheckSize(y, yLabel);
    
    // any of the following cases is guaranteed by the constructor
    if (Name() == "mse") { 
        *(_gradient) = (2/double(y.size()))*(y-yLabel).transpose();
    }
}


