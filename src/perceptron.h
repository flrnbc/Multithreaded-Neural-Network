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
    std::vector<double> _bias;
    // perceptron data
    std::unique_ptr<PerceptronData> _data;
    // activation
    std::unique_ptr<ActivationFct> _activationFct;

public:
    // (default) constructor
    Perceptron(std::vector<std::vector<double> > weights, std::vector<double> bias, std::string fct);
    // constructor using random initialization of weights
    Perceptron(int rows, int cols, std::string fct);

    // destructor
    // TODO #A: issue with empty destructor?
    ~Perceptron();

    // copy constructor
    Perceptron(const Perceptron&);

    // copy assignment
    Perceptron& operator=(const Perceptron&);

    // setters & getters for perceptron data
    std::vector< std::vector<double> > Weights() { return _weights; }
    void SetWeights(std::vector<std::vector<double> > weight_matrix) {
        _weights = weight_matrix;
    }
    std::vector<double> Bias() { return _bias; }
    void SetBias(std::vector<double> bias) { _bias = bias; }
    int Rows(); 
    int Cols(); 
    std::string Activation();

    // setters & getters for activation function
    //void SetActivationFct(std::string fct); 

    // evaluate on a vector
    std::vector<double> Evaluate(std::vector<double>);

    // summary
    std::string Summary();
};

#endif // PERCEPTRON_H_
