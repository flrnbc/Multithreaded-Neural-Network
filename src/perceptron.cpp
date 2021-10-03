#include <random>
#include <stdexcept>
#include <string>
#include <vector>

//#include "perceptron.h"



// Perceptron PerceptronData::Initialize() {
//     // random initialization
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     // 'quasi-linear' activation functions use different random initialization
//     std::string quasiLinear [] = {"id", "sigmoid", "tanh"};
//     // TODO #A: relu (and possibly prelu) now have the same initialization as heaviside which might not be ideal

//     // perceptron data
//     int cols = this->Cols();
//     int rows = this->Rows();    
//     std::vector< std::vector<double> > weights;
//     double bias;

//     // initialize weights
//     if (std::find(std::begin(quasiLinear), std::end(quasiLinear), this->Activation()) != std::end(quasiLinear)) {
//         // use normalized Xavier weight initialization
//         // TODO #A: give reference
//         int inputPlusOutputSize = cols + rows;
//         // NOTE: uniform_real_distribution(a, b) generates for [a, b) (half-open interval)
//         std::uniform_real_distribution<> dis(-(sqrt(6)/sqrt(inputPlusOutputSize)), sqrt(6)/sqrt(inputPlusOutputSize));

//         // randomly initialize weights
//         // NOTE: be careful with cache-friendliness (outer loop over rows)
//         // TODO #A: would be nicer to put this for-loop after the else-block (but then would have to declare dis before; but as what?)
//         for (int i=0; i<rows; i++) {
//             for (int j=0; j<cols; j++) {
//                 weights[i][j] = dis(gen); 
//             }
//         }
//         // initialize bias
//         bias = dis(gen);
//     } 
//     else {
//         // use He weight initialization
//         // TODO #A: give reference
//         std::normal_distribution<> dis(0.0, sqrt(2/cols));

//         // randomly initialize weights
//         for (int i=0; i<rows; i++) {
//             for (int j=0; j<cols; j++) {
//                 weights[i][j] = dis(gen); 
//             }
//         }
//         // initialize bias
//         bias = dis(gen);
//     }
 
//     return Perceptron(weights, bias, this->Activation());
// }


// Perceptron::Perceptron(std::vector<std::vector<double> > weights, double bias, std::string activation) {
//     SetWeights(weights);
//     SetBias(bias);
//     // initialize perceptron data
//     // TODO: test if it rejects 'empty vectors'
//     _data.SetActivation(activation);
//     _data.SetRows(weights.size());
//     _data.SetCols(weights[0].size());
// }


// std::vector<double> Perceptron::Evaluate(std::vector<double> inputVector) {
//     std::vector<std::vector<double> > weights = this->Weights();
//     std::vector<double> outputVector;
//     double bias = this->Bias();

//     for (int i=0; i < this->Rows(); i++) {
//         // matrix multiplication weights*inputVector
//         for (int j=0; j < this->Cols(); j++) {
//             outputVector[i] += weights[i][j]*inputVector[j];
//         }
//         // add bias 
//         outputVector[i] += bias;
//     }

//    return outputVector;
//}