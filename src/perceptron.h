#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_

#include <string>
#include <vector>

//#include "activation.h"
//#include "perceptron_data.h"

//class PerceptronData;
class ActivationFct;

class Perceptron
{
private:
    std::vector<std::vector<double> > _weights;
    double _bias;
    // perceptron data
    int _numberOfRows;
    int _numberOfCols;
    std::string _activation;
    // activation
    ActivationFct _activationFct;

public:
    // constructor
    Perceptron(std::vector<std::vector<double> > weights, double bias, std::string fct);
        //: _weights(weights), _bias(bias), _data(weights.size(), weights[0].size(), fct), 
        //_activationFct(ActivationFct(fct))  
        
    //{
    //}

    // setters & getters for perceptron data
    std::vector< std::vector<double> > Weights() { return _weights; }
    void SetWeights(std::vector<std::vector<double> > weight_matrix) {
        _weights = weight_matrix;
    }
    double Bias() { return _bias; }
    void SetBias(double bias) { _bias = bias; }
    int Rows() { return _numberOfRows; }
    void SetRows(int rows) { _numberOfRows = rows; }
    int Cols() { return _numberOfCols; }
    void SetCols(int cols) { _numberOfCols = cols; }
    std::string Activation() { return _activation; }
    void SetActivation(std::string activation) { _activation = activation; }

    // setters & getters for activation function
    //ActivationFct GetActivationFct() { return _activationFct; } 
    void SetActivationFct(std::string fct) {
        _activationFct = ActivationFct(fct);
    }

    // evaluate on a vector
    std::vector<double> Evaluate(std::vector<double>);
};

#endif // PERCEPTRON_H_
