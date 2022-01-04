#ifndef _LOSS_FCT_H_
#define _LOSS_FCT_H_

#include <Eigen/Dense>
#include "function.h"
#include <string>


class LossFunction {
    private:
    // input dimension
    int _cols;
    // name of loss function
    std::string _name;
    // gradient
    std::unique_ptr<Eigen::RowVectorXd> _gradient;

    public: 
        // constructor
        LossFunction(std::string name, int cols); 

        // getters
        int Cols() { return _cols; }
        std::string Name() { return _name; }
        Eigen::RowVectorXd& Gradient() { return *(_gradient); }

        // compute loss
        double ComputeLoss(Eigen::VectorXd&, const Eigen::VectorXd&);

        // update derivative/gradient (see below)
        void UpdateGradient(Eigen::VectorXd&, const Eigen::VectorXd&);
};


#endif // _LOSS_FCT_H_