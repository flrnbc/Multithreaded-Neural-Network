
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>
#include "../src/transformation.h"
#include "../src/layer.h"
#include "../src/layer_cache.h"
#include "../src/loss_function.h"

/**
 * Smoke tests for Layer class.
 */

void test_LinearLayer() {
    LinearLayer ll(5, 3);
    std::cout << ll.Summary() << std::endl;
    Eigen::MatrixXd X(3, 4);
    X << 1, 0, 0, 0,
         0, 2, 3, 10,
         9, 0, 1, 5;
    Eigen::MatrixXd W(4, 5);
    W << 1, 0, 0, 0, 1,
         0, 2, 6, 2, 7, 
         3, 1, 5, 9, 8,
         0, 2, 6, 2, 0;

    ll.GetTransformation()->Initialize("Xavier"); 
    std::cout << ll.Summary() << std::endl;

    ll.Input(X);
    ll.Forward();
    std::cout << "Transform X: \n" << ll.Output() <<  std::endl;

    // test UpdateWeights
    ll.BackwardInput(W);
    ll.UpdateWeightsBias(0.1);
    
    std::cout << "After updating weights and bias: " << ll.Summary() << std::endl;
}

void test_LinearLayer2() {
    LinearLayer ll(1, 1);
    auto mse = LossFunction("mse");
    //mse.SetCols(1);

    ll.GetTransformation()->Initialize("Xavier");
    std::cout << ll.Summary() << std::endl;

    // training test
    Eigen::MatrixXd x{{3}};
    Eigen::MatrixXd yLabel{{2}};

    ll.Input(x);
    ll.Forward();
    //ll.Derivative();

    Eigen::MatrixXd y = ll.Output();
    std::cout << "Output: " << y << std::endl;
    std::cout << "Loss: " << mse(y, yLabel) << std::endl;

    mse.GradsAtPoints(y, yLabel);
    std::cout << "Gradient of mse: " << mse.GetGrads() << std::endl;

    ll.BackwardInput(mse.GetGrads());
    ll.Backward();
    ll.UpdateWeightsBias(0.1);

    std::cout << "Summary after training: " << ll.Summary() << std::endl;
}

void test_ActivationLayer() {
    ActivationLayer al(8, "softmax");
    std::cout << al.Summary() << std::endl;

    Eigen::VectorXd v{{1, -5, 0, 1, 3, -6, -8, 0}};
    Eigen::RowVectorXd w{{1, 0, 1, 0, 1, 0, 1, 0}};

    // forward pass
    al.Input(v);
    al.Forward();

    std::cout << "Transform v: \n" << al.Output() << std::endl;

    // update derivative
    //al.Derivative();
    //al.GetTransformation()->Derivative(al.Output());
    std::cout << "After update: \n" << al.Summary() << std::endl;

    // backward pass
    al.BackwardInput(w);
    al.Backward();

    std::cout << "Backward output: \n" << al.BackwardOutput() << std::endl;
}

void test_LayerVector() {
    std::vector<Layer> v{{LinearLayer(4, 8), ActivationLayer(4, "softmax")}};
    std::vector<std::shared_ptr<Layer> > w;

    for (int i=0; i<2; i++) {
        w.emplace_back(std::make_shared<Layer>(v[i]));
    }

    std::cout << w[1]->Summary() << std::endl;
    std::cout << "Count of objects: " << w[1].use_count() << std::endl;
}

int main() {
    test_LinearLayer();
    //test_ActivationLayer();
    //test_LayerVector();
}