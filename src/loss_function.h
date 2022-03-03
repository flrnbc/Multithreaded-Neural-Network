#ifndef _LOSS_FCT_H_
#define _LOSS_FCT_H_

#include <Eigen/Dense>
#include "function.h"
#include <memory>
#include <stdexcept>
#include <string>

/*****************
 * LOSS FUNCTION *
 *****************/

/** 
    Class encoding loss functions (vectors -> double) and their derivatives.
*/

class LossFunction {
    private:
    // name of loss function
    std::string _name;
    // gradients (row vectors)
    // We work with 'linear loss functions' (i.e. the loss of a batch is the sum of the 
    // losses of each data sample in the batch). Then it is convenient to store the gradients
    // for each data sample in a matrix.
    Eigen::MatrixXd _gradients; 

    public: 
        // constructor
        LossFunction(std::string name);
        
        // getters
        std::string Name() { return _name; }
        Eigen::MatrixXd& GetGrads() { 
            return _gradients; 
        }

        // check if sizes of input vectors (columns of the two matrices) coincide
        void CheckSize(Eigen::MatrixXd&, const Eigen::MatrixXd&);

        // compute total loss of a batch (sum of the losses of each data sample in the batch)
        double operator()(Eigen::MatrixXd&, const Eigen::MatrixXd&);

        // update derivative/gradient 
        void GradsAtPoints(Eigen::MatrixXd&, const Eigen::MatrixXd&);

        // set gradients to zero
        void ZeroGrads();
};

#endif // _LOSS_FCT_H_