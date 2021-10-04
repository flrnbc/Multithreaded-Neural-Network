#include <iostream>
#include <string>
#include <vector>

#include "activation.h"
#include "perceptron_data.h"

void test_PerceptronData() {
    PerceptronData pd = PerceptronData(2, 2, "relu");
    PerceptronData pd2;
    
    // testing 'Activation part'
    std::cout << pd.Activation() << std::endl;
    pd.SetActivation("identity");
    std::cout << pd.Activation() << std::endl;
    std::cout << pd2.Activation() << std::endl;

    // testing Rows and Cols
    std::cout << pd.Rows() << std::endl;
    std::cout << pd.Cols() << std::endl;
    std::cout << pd2.Rows() << std::endl;
    pd.SetCols(0);
    pd.SetRows(0);
}


void test_activationFcts() {
    std::cout << "heaviside(2.0) = " << heaviside(2.0) << std::endl;
    std::cout << "relu(2.0) = " << relu(2.0) << std::endl;
    std::cout << "sigmoid(2.0) = " << sigmoid(2.0) << std::endl;
    std::cout << "tanh(2.0) = " << tanh(2.0) << std::endl;
}


void test_ActivationFct() {
    ActivationFct actFctId = ActivationFct("identity");
    ActivationFct actFctHeaviside = ActivationFct("heaviside");
    ActivationFct actFctRelu = ActivationFct("relu");
    ActivationFct actFctSigmoid = ActivationFct("sigmoid");
    ActivationFct actFctTanh = ActivationFct("tanh");
    //ActivationFct actWrong = ActivationFct("fantasy");

    std::vector<double> input{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};

    std::cout << actFctId.Name() << std::endl;
    for (double d: actFctId.Evaluate(input)) {
        std::cout << d << std::endl;
    }

    std::cout << actFctHeaviside.Name() << std::endl;
    for (double d: actFctHeaviside.Evaluate(input)) {
        std::cout << d << std::endl;
    }

    std::cout << actFctRelu.Name() << std::endl;
    for (double d: actFctRelu.Evaluate(input)) {
        std::cout << d << std::endl;
    }

    std::cout << actFctSigmoid.Name() << std::endl;
    for (double d: actFctSigmoid.Evaluate(input)) {
        std::cout << d << std::endl;
    }

    std::cout << actFctTanh.Name() << std::endl;
    for (double d: actFctTanh.Evaluate(input)) {
        std::cout << d << std::endl;
    }

}


void test_Perceptron() {
    std::vector<std::vector<double> > weights{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    double bias = 5.0;
    Perceptron per(weights, bias, "sigmoid");
    std::vector<double> input{-1, 0, 1};

    //per.Evaluate(input);
    for (double d: per.Evaluate(input)) {
        std::cout << d << std::endl;
    }
}


int main() {
    //test_PerceptronData();
    //test_activationFcts();
    //test_ActivationFct();
    test_Perceptron();

    return 0;
}