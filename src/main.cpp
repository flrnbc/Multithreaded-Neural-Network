#include <iostream>
//#include <vector>

//#include "activation.h"
#include "perceptron_data.h"
//#include "tests.h"

int main() {
    PerceptronData pd = PerceptronData(2, 2, "relu");
    
    //std::cout << pd.Activation() << std::endl;
    std::cout << pd.Rows() << std::endl;
    std::cout << pd.Cols() << std::endl;   

    return 0;
}