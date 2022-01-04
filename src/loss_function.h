#ifndef _LOSS_FCT_H_
#define _LOSS_FCT_H_

#include <Eigen/Dense>
#include "function.h"
#include <string>

/***********************
 * ABSTRACT BASE CLASS *
 ***********************/

class LossFunction {
    protected:
    // input dimension
    int _cols;
    // name of loss function
    std::string _name;
    // gradient
    std::unique_ptr<Eigen::RowVectorXd> _gradient;

    LossFunction(int cols, std::string name):
        _cols(cols),
        _name(name),
        _gradient(std::make_unique<Eigen::RowVectorXd>(Eigen::RowVectorXd::Zero(cols)))
        {}

    public: 
        // virtual destructor
        virtual ~LossFunction() {}

        // getters
        int Cols() { return _cols; }
        std::string Name() { return _name; }
        Eigen::RowVectorXd& Gradient() { return *(_gradient); }

        // compute loss
        virtual double ComputeLoss(Eigen::VectorXd&, const Eigen::VectorXd&) = 0;

        // update derivative/gradient (see below)
        virtual void UpdateGradient(Eigen::VectorXd&, const Eigen::VectorXd&) = 0;
};

class MSE: public LossFunction {
    public: 
        MSE(int cols):
            LossFunction(cols, "Mean error square")
            {}
        
        // compute loss for pair (y, yLabel) where y is the prediction and yLabel
        double ComputeLoss(Eigen::VectorXd&, const Eigen::VectorXd&);

        // update derivative/gradient at (y, yLabel)
        void UpdateGradient(Eigen::VectorXd&, const Eigen::VectorXd&);
};


#endif // _LOSS_FCT_H_