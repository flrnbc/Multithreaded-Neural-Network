#include "loss_function.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

/*******
 * MSE *
 *******/

double MSE::ComputeLoss(Eigen::VectorXd& y, const Eigen::VectorXd& yLabel) {
    if (y.size() != yLabel.size()) {
        throw std::invalid_argument("Vectors are not of same size.");
    }
    Eigen::VectorXd e = y-yLabel;
    double avg = 1/double(y.size());
    double scalar_prod = (e.transpose()*e)(0);

    return avg*scalar_prod;
}

void MSE::UpdateGradient(Eigen::VectorXd& y, const Eigen::VectorXd& yLabel) {
    if (y.size() != yLabel.size()) {
        throw std::invalid_argument("Vectors are not of same size.");
    }
    *(_gradient) = (2/double(y.size()))*(y-yLabel).transpose();
}