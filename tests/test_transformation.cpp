#include <Eigen/Dense>
#include <iostream>
#include "../src/transformation.h"

void test_RandomWeightInitialization() {
    auto t = LinearTransformation(3, 3);
    t.Initialize("He");
    auto t2 = LinearTransformation(3, 6);
    t2.Initialize("Xavier");

    std::cout << "Transformation t: " << std::endl;
    std::cout << t.Summary() << std::endl;

    std::cout << "Transformation t2: " << std::endl;
    std::cout << t2.Summary() << std::endl;
}

void test_LinearTransform() {
    auto t = LinearTransformation(3, 6);
    t.Initialize("He");
    Eigen::VectorXd e1{{1.0, 0, 0, 0, 0, 0}};
    Eigen::VectorXd e2{{0, 1.0, 0, 0, 0, 0}};

    std::cout << "Transformation t: " << std::endl;
    t.Derivative(e1);
    std::cout << t.Summary() << std::endl;

    std::cout << "Transform e1: " << std::endl;
    std::cout << t.Transform(e1) << std::endl;

    std::cout << "Transform e2: " << std::endl;
    std::cout << t.Transform(e2) << std::endl;

    // check if derivative is correctly updated
    auto random_matrix = Eigen::MatrixXd::Random(3, 6);
    t.SetWeights(random_matrix);
    t.Derivative(e1);
    std::cout << "After updating weights: \n" << t.Summary() << std::endl;

    // update delta
    Eigen::RowVectorXd w{{1, 0, 0}};
    std::cout << "Update Delta:" << t.BackwardTransform(w) << std::endl;
}

// tests for ActivationTransformation
// TODO: use reference because copy constructor implicitly deleted (might have to change that)
void test_UpdateDelta(ActivationTransformation& t, Eigen::RowVectorXd delta) {
    std::cout << "Updated Delta: \n" << t.BackwardTransform(delta) << std::endl;
}

void test_ActivationTransformation(ActivationTransformation& a, Eigen::VectorXd vector, Eigen::RowVectorXd delta) {
    std::cout << "Initial summary \n" << a.Summary() << std::endl;
    std::cout << "Transform vector: " << std::endl;
    std::cout << a.Transform(vector) << std::endl;

    // update derivative
    a.Derivative(a.Transform(vector));
    std::cout << "After updating derivative \n" << a.Summary() << std::endl;

    // update delta
    test_UpdateDelta(a, delta);
}

void test_ActivationTransforms() {
    Eigen::VectorXd e1{{10.0, 0.5, 2.0, 1.0, 0}};
    Eigen::VectorXd vector{{2.0, 5.0, -1.0, -1.0, 0}};

    auto a = ActivationTransformation(5, "relu");
    auto a2 = ActivationTransformation(5, "sigmoid");
    auto a3 = ActivationTransformation(5, "tanh");
    auto a4 = ActivationTransformation(5, "softmax");

    test_ActivationTransformation(a, vector, e1);
    test_ActivationTransformation(a2, vector, e1);
    test_ActivationTransformation(a3, vector, e1);
    test_ActivationTransformation(a4, vector, e1);
}


int main() {
    //test_RandomWeightInitialization();
    test_LinearTransform();

    //test_ActivationTransforms();
}