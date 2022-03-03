# Sequential neural networks in `C++17` (almost) from scratch

The aim of this project is to implement a basic but flexible API in modern `C++` to build and train sequential neural networks[^1], i.e. neural networks whose hidden layers have exactly one input and one output layer. It is my submission for the Capstone project[^2] in the [Udacity C++ Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213). To showcase the API, we train a neural network with the MNIST dataset (handwritten digits between 0 and 9, e.g. see [this link](http://yann.lecun.com/exdb/mnist/)). Unfortunately, the performance is very poor at this point[^3]. However, the main point of this project is *not* to create a competitor of the well established neural network APIs (e.g. `tensorflow` or `keras`). Instead the purpose is to apply techniques from (modern) `C++` like smart pointers, move semantics, templates, libraries and abstract classes. The latter allows the user to easily add new types of Layers etc.

At the beginning, I tried to do everything from scratch but then quickly realized that would be too much. For example, it would have required the implementation of a fully functioning matrix class (with matrix multiplication etc.). Hence the only external dependency is the excellent [Eigen library](https://eigen.tuxfamily.org/) which deals with the necessary matrix calculus.  
>>>>>>> MiniBatch

## Instantiating a (sequential) neural network
We provide two types of layers, a `LinearLayer` and an `ActivationLayer`. They just encapsulate an affine-linear transformation and activation function respectively. To instantiate a sequential neural network with just a `LinearLayer` followed by an `ActivationLayer` with the ReLu activation function (so we create a [Percpetron](https://en.wikipedia.org/wiki/Perceptron)), we use the following code. The includes should be the same as in [main.cpp](../main.cpp) and are omitted for clarity:

```C++
int inputSize = 20; // number of columns of the matrix in the LinearLayer
int outputSize = 10; // number of rows of the matrix in the LinearLayer
auto perceptron = SequentialNN({LinearLayer(outputSize, inputSize), ActivationLayer(outputSize, "relu")});
// evaluate on an Eigen matrix with random entries
int batchSize = 4;
Eigen::MatrixXd X = Eigen::MatrixXd::Random(inputSize, batchSize);
std::cout << "Output: \n" << snn(X) << std::endl;
```

Clearly, we can add as many layers as we want. The only condition they have to satisfy is that the input size of the i-th layer coincides with the output size of the (i-1)-th layer. 

NOTE:
+ We work with number of rows and columns so that in a `LinearLayer` the output size comes first. Since an `ActivationLayer` does not change the input size, we only need to give the input size in the constructor.
+ The weights in the `LinearLayer` are initialized according to the so-called He and Xavier intialization (see e.g. [this link](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/)). 

## Training a neural network with mini-batch learning (stochastic gradient descent)
With the `DataParser` (template) class we can load data from a `.csv`-file, say `train_samples.csv` and `train_labels.csv`, which already contain the train data split into data samples and their labels [^4]. Then we train our perceptron with the mean squarred error (MSE) as loss function, `batchSize` and `learningRate` via (continuation from above):

```C++
DataParser dp;
Eigen::MatrixXd trainSamples=dp.LoadCSV<Eigen::MatrixXd>(train_samples.csv);
Eigen::MatrixXd trainLabels=dp.LoadCSV<Eigen::MatrixXd>(train_labels.csv);
int batchSize=10;
double learningRate=0.001;
auto sdg=SDG("mse", batchSize, learningRate); // mini-batch stochastic gradient descent with MSE as loss function
// actual training
int epochs=1000; // number of epochs in the training
sdg.Train(snn, trainSamples, trainLabels, epochs);
```

That's it! Now we can apply our `snn` to some test data and evaluate it. This is not yet fully implemented since one might have to encode the original data labels (e.g. via one-hot-encoding) first. However, we provide an evaluation in [main.cpp](../main.cpp) for MNIST.

## Compilation
With `CMake` we can simply build `main.cpp` via 

```bash
cmake --build /Users/fbeck/Documents/Rise/C++/udacity/CapstoneProject/CppND-Capstone-Hello-World/build --config Debug --target Main
```

## Tests
For several tests, see the `tests` folder. These are quite primitive tests, so-called 'smoke tests', and do not use any test suite. In case, one is interested, one can easily build each of them via

```bash
cmake --build /Users/fbeck/Documents/Rise/C++/udacity/CapstoneProject/CppND-Capstone-Hello-World/build --config Debug --target TestName
```

Here `TestName` is either `TestLossFunction`, `TestLayer` etc. `TestOptimizer` might be the most interesting because it contains a simple example (1-dimensional linear regression in the function `test_OptimizeLinearRegression1D()`) where this API does perform quite well.

[^1]: This terminology might be unconvential but comes from `Sequential` models of `Keras`. 
[^2]: The nice thing about Udacity Nanodegrees is that the final (Capstone) projects are entirely up to the student. 
[^3]: There might be an issue with vanishing gradients of the softmax activation function. Or even some tricky mistake in the implementation of the backpropagation algorithm.
[^4]: See [main.cpp](../main.cpp) for a simple function which does this splitting in the special case of MNIST.
