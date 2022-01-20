#ifndef _LOSS_FCT_H_
#define _LOSS_FCT_H_

#include <Eigen/Dense>
#include "function.h"
#include <memory>
#include <stdexcept>
#include <string>


class LossFunction {
    private:
    // name of loss function
    std::string _name;
    // input size
    int _cols;
    // gradient
    std::unique_ptr<Eigen::RowVectorXd> _gradient = nullptr;

    public: 
        // constructor
        LossFunction(std::string name); 

        // setters
        void SetCols(int cols);
        
        // getters
        int Cols() { return _cols; }
        std::string Name() { return _name; }
        Eigen::RowVectorXd& Gradient() { 
            if (_gradient == nullptr) {
                throw std::invalid_argument("Gradient pointer is null.");
            }
            return *(_gradient); 
        }

        // check size of input vectors
        void CheckSize(Eigen::VectorXd&, const Eigen::VectorXd&);

        // compute loss
        double operator()(Eigen::VectorXd&, const Eigen::VectorXd&);

        // update derivative/gradient
        void UpdateGradient(Eigen::VectorXd&, const Eigen::VectorXd&);
};


#endif // _LOSS_FCT_H_