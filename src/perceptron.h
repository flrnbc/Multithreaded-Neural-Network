#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_

#include <memory>
#include <string>
#include <vector>

//#include "activation.h"
//#include "perceptron_data.h"

class ActivationFct;
class PerceptronData;

class Perceptron
{
private:
    std::vector<std::vector<double> > _weights;
    double _bias;
    // perceptron data
    std::unique_ptr<PerceptronData> _data;
    // activation
    std::unique_ptr<ActivationFct> _activationFct;

public:
    // constructor
    Perceptron(std::vector<std::vector<double> > weights, double bias, std::string fct);
    
    // setters & getters for perceptron data
    std::vector< std::vector<double> > Weights() { return _weights; }
    void SetWeights(std::vector<std::vector<double> > weight_matrix) {
        _weights = weight_matrix;
    }
    double Bias() { return _bias; }
    void SetBias(double bias) { _bias = bias; }
    int Rows(); 
    int Cols(); 
    std::string Activation();

    // setters & getters for activation function
    //void SetActivationFct(std::string fct); 

    // evaluate on a vector
    std::vector<double> Evaluate(std::vector<double>);
};

#endif // PERCEPTRON_H_
