#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "layer.h"
#include "layer_cache.h"
#include "transformation.h"
#include "sequential_nn.h"

/*******************
 * HELPER FUNCTION *
 *******************/

bool ComposabilityCheck(std::vector<std::unique_ptr<Layer> > layers) {
    bool composable = true;
    int N = layers.size();

    for (int i=0; i<N-1; i++) {
        if (layers[i]->Cols() != layers[i+1]->Rows()) {
            composable = false;
            break;
        }
    } 
    return composable;
}


/********************************************
 * SEQUENTIAL NEURAL NETWORK IMPLEMENTATION *
 ********************************************/

SequentialNN::SequentialNN(std::vector<std::unique_ptr<Layer> > layers) {
    if (ComposabilityCheck(layers) == false) {
        throw std::invalid_argument("Layers are not composable.");
    } else {
        int N = layers.size();

        for (int i = 0; i < N-1; i++) {
            _layers[i]->GetLayerCache()->ConnectForward(_layers[i]->Rows(), *(_layers[i+1]->GetLayerCache()));
        }
    }
}

// std::string SequentialNN::Summary() {
//     std::string summary = "Summary of sequential neural network: ";
//     int N = GetLayers().size();

//     for (int i=0; i < N; i++) {
//         summary += "\nLayer " + std::to_string(i) + ":\n" + _layers[i].Summary();
//     }

//     return summary;
// }

void SequentialNN::Forward() {
    for (int i=0; i < Length(); i++) {
        _layers[i]->Forward();
        //std::cout << "input (in Forward): " << Perceptron::PrintDoubleVector(_pointers_layers[i]->InputData()) << std::endl;
        //std::cout << "output (in Forward): " << Perceptron::PrintDoubleVector(_pointers_layers[i]->OutputData()) << std::endl;
    }
}

void SequentialNN::Input(std::vector<double> input) {
    _layers[0]->Input(input);
}