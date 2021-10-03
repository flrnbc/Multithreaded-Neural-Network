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

void test_ActivationFct() {
    
    std::cout << "heaviside(2.0) = " << heaviside(2.0) << std::endl;
    std::cout << "relu(2.0) = " << relu(2.0) << std::endl;
    std::cout << "sigmoid(2.0) = " << sigmoid(2.0) << std::endl;
    std::cout << "tanh(2.0) = " << tanh(2.0) << std::endl;


    ActivationFct actFctId = ActivationFct("identity");
    std::cout << actFctId.Name() << std::endl;


    ActivationFct actFctHeaviside = ActivationFct("heaviside");
    ActivationFct actFctRelu = ActivationFct("relu");
    ActivationFct actFctSigmoid = ActivationFct("sigmoid");
    ActivationFct actFctTanh = ActivationFct("tanh");
    //ActivationFct actWrong = ActivationFct("fantasy");

    std::vector<double> input {1, 1, 1, 2};

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



int main() {
    //test_PerceptronData();
    test_ActivationFct();

    return 0;
}