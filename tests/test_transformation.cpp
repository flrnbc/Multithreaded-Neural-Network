#include <iostream>
#include "../src/transformation.h"

void test_RandomWeightInitialization() {
    LinearTransformation t = LinearTransformation(3, 3);
    t.Initialize("He");
    auto t2 = LinearTransformation(3, 6);
    t2.Initialize("Xavier");

    std::cout << "Transformation t: " << std::endl;
    std::cout << t.Summary() << std::endl;

    std::cout << "Transformation t2: " << std::endl;
    std::cout << t2.Summary() << std::endl;
}

void test_LinearTransform () {
    auto t = LinearTransformation(3, 6);
    t.Initialize("He");
    std::vector<double> e1{1.0, 0, 0, 0, 0, 0};
    std::vector<double> e2{0, 1.0, 0, 0, 0, 0};

    std::cout << "Transformation t: " << std::endl;
    std::cout << t.Summary() << std::endl;

    std::cout << "Transform e1: " << std::endl;
    for (int i=0; i<3; i++) {
        std::cout << t.Transform(e1)[i] << std::endl;
    }

    std::cout << "Transform e2: " << std::endl;
    for (int i=0; i<3; i++) {
        std::cout << t.Transform(e2)[i] << std::endl;
    }
}

void test_ActivationTransform () {
    auto a = ActivationTransformation(5, "relu");
    std::vector<double> e1{1.0, 0, 0, 0, 0};
    std::vector<double> e2{0, 1.0, 0, 0, 0};

    std::cout << "Transformation a: " << std::endl;
    std::cout << a.Summary() << std::endl;

    std::cout << "Transform e2: " << std::endl;
    for (int i=0; i<5; i++) {
        std::cout << a.Transform(e2)[i] << std::endl;
    }

}

int main() {
    test_RandomWeightInitialization();
    test_LinearTransform();
    test_ActivationTransform();
}