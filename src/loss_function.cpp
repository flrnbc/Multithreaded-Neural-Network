#include "loss_function.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// constructor
std::vector<std::string> loss_functions = {"mse"};

LossFunction::LossFunction(std::string name, int cols) {
    auto find_loss_fct = std::find(std::begin(loss_functions), std::end(loss_functions), name);

    if (find_loss_fct == loss_functions.end()) {
        throw std::invalid_argument("Unknown loss function.");
    }

    _cols = cols;
    _name = name;
    _gradient = std::make_unique<Eigen::RowVectorXd>(Eigen::RowVectorXd::Zero(cols));
}

// methods
double LossFunction::ComputeLoss(Eigen::VectorXd& y, const Eigen::VectorXd& yLabel) {
    if (y.size() != yLabel.size()) {
        throw std::invalid_argument("Vectors are not of same size.");
    }
  
    // any of the following cases is guaranteed by the constructor
    if (Name() == "mse") {
        Eigen::VectorXd e = y-yLabel;
        double avg = 1/double(y.size());
        double scalar_prod = (e.transpose()*e)(0);

        return avg*scalar_prod;
    }
}

void LossFunction::UpdateGradient(Eigen::VectorXd& y, const Eigen::VectorXd& yLabel) {
    if (y.size() != yLabel.size()) {
        throw std::invalid_argument("Vectors are not of same size.");
    }
    
    // any of the following cases is guaranteed by the constructor
    if (Name() == "mse") { 
        *(_gradient) = (2/double(y.size()))*(y-yLabel).transpose();
    }
}