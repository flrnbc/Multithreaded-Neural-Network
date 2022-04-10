# Sequential neural networks in `C++17` (almost) from scratch

The aim of this project is to implement a basic but flexible API in modern `C++` to build and train sequential neural networks[^1], i.e. neural networks whose hidden layers have exactly one input and one output layer. It is my submission for the Capstone project[^2] in the [Udacity C++ Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213). To showcase the API, we train a neural network with the MNIST dataset (handwritten digits between 0 and 9, e.g. see [this link](http://yann.lecun.com/exdb/mnist/)). Unfortunately, the performance is very poor at this point[^3]. However, the main point of this project is *not* to create a competitor of the well established neural network APIs (e.g. `tensorflow` or `keras`). Instead the purpose is to apply techniques from (modern) `C++` like smart pointers, move semantics, templates, libraries and abstract classes. The latter allows the user to easily add new types of Layers etc.

At the beginning, I tried to do everything from scratch but then quickly realized that would be too much. For example, it would have required the implementation of a fully functioning matrix class (with matrix multiplication etc.). Hence the only external dependency is the excellent [Eigen library](https://eigen.tuxfamily.org/) which deals with the necessary matrix calculus. It is contained in this repository as a [git submodule][https://git-scm.com/book/en/v2/Git-Tools-Submodules].

## Instantiating a (sequential) neural network
We provide two types of layers, a `LinearLayer` and an `ActivationLayer`. They just encapsulate an affine-linear transformation and activation function respectively. To instantiate a sequential neural network with just a `LinearLayer` followed by an `ActivationLayer` with the ReLu activation function (so we create a [Percpetron](https://en.wikipedia.org/wiki/Perceptron)), we use the following code. The includes should be the same as in [main.cpp](main.cpp) and are omitted for clarity:

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

That's it! Now we can apply our `snn` to some test data and evaluate it. This is not yet fully implemented since one might have to encode the original data labels (e.g. via one-hot-encoding) first. However, we provide an evaluation in [main.cpp](main.cpp) for MNIST.

## Installation/Compilation
As noted above, the only dependency `eigen` is contained as a git submodule. Hence clone this repository with this submodule via
```bash
git clone --recurse-submodules https://github.com/flrnbc/Sequential-Neural-Networks
```
Next we change to the project root directory, initialize `cmake` and create a `build` directory via
```bash
cd Sequential-Neural-Networks
mkdir build/
```
Now we build `main.cpp` via 
```bash
cd build
cmake .. 
cmake --build . --target Main
```
It is executed via 
```bash
cd ..
build/Main
```
(TODO: changing back to the project root is still necessary at the moment but will be fixed.)


### Building tests
The `cpp`-files in `tests` contain all the (mainly smoke) tests for each class. Please take a look at these files to activate the test functions you are interested in. Then a test is built and run via (after `cmake` has already been run as above)
```bash
cd build/
cmake --build .. --target TestName // where Name is either DataParser, Function, LayerCache, Layer, LossFct, Optimizer, SequentialNN or Transformation
cd ..
build/TestName // again replace Name correspondingly
```
We recommend trying `TestOptimizer` first (with the function `test_OptimizeLinearRegression1D`) since it showcases the training of the simplest neural network possible.


# Project file and class structure
## File structure
The source code is contained in the `src` directory. Typically, each `.h/.cpp` file pair declares/defines one class, e.g. the `Layer` class is declared/defined in `layer.h` and `layer.cpp`. All tests are contained in the `tests` directory, e.g. `tests/test_layer.cpp` contains tests for the `Layer` class. Finally, the `Eigen` library is in the `eigen` directory.

## Relation between classes
For more information on the classes, please see the corresponding header files.

+ `Transformation`: abstract class with `LinearTransformation` and `ActivationTransformation` as concrete derived classes. The latter uses the `Function` class and both use the `Eigen::Matrix` class.
+ `Layer`: abstract class with `LinearLayer` and `ActivationLayer` as concrete derived classes. Built from `LinearTransformation` and `ActivationTransformation` respectively as well as the `LayerCache` class.
+ `SequentialNN`: built from a vector `Layer` classes.
+ `Optimizer`: smart pointer to `LossFunction` object as member variable. 
+ `DataParser`: uses the `Eigen::Matrix` class.


# Project requirements
The project needs to fulfill several requirements to successfully pass the Capstone review process. Here we give the positions in our code where the respective requirement is satisfied. 

## Loops, Functions, I/O
+ *Demonstrate understanding of C++ functions and control structures*: 
  For example see [src/layer.cpp](src/layer.cpp) in the function `Layer::Forward()` (line 34 ff.) where conditionals are used to handle an exception.
+ *Reading data from a file and processing it*: 
  This is fulfilled in [src/data_parser.h](src/data_parser.h) in the function `LoadCSV` (see line 30 ff.) which reads data from a `.csv`-file and saes it in an `Eigen::Matrix` object.

## Object Oriented Programming (OOP)
+ *Using OOP techniques*: 
  The design of the sequential neural network API relies heavily on OOP methods. As an example, see the class `SequentialNN` in [src/sequential_nn.h](src/sequential_nn.h), line 55 ff., which has several member variables and functions.
+ *Class access specifiers for class members*:
  Used several times, for example in [src/transformation.h](src/transformation.h), lines 49 - 94. There we use `protected` to facilitate the access to member variables for derived classes.
+ *Class constructors utilize member initialization lists*: 
  Applied to several constructors (where appropriate), e.g. in [src/layer.h](src/layer.h), lines 125 - 134.
+ *Classes follow an appropriate inheritance hierarchy*:
  Multiple classes inherit from virtual base classes, for example the `LinearTransformation` class from `Transformation` in [src/transformation.h](src/transformation.h), line 101 ff. Moreover, composition is used (albeit via smart pointers) e.g. in the `Layer` class, see [src/layer.h](src/layer.h), line 47 ff.
+ *Derived class functions override virtual base class functions*: 
  This is done, for example, in the function `ZeroDeltaWeights` of the `LinearLayer` class, see [src/layer.h](src/layer.h), line 120.
+ *Templates generalize functions*:
  See the function `LoadCSV` in [src/data_parser.h](src/data_parser.h), line 31.

## Memory Mangement
+ *Rule of 5*: 
  See [src/layer_cache.cpp](src/layer_cache.cpp), line 56 ff., even though we use the default move constructors/assignment operator (which seems to be ok because `LayerCache` is composed of member variables which admit move semantics).
+ *Using move semantics*: 
  Applied in `LayerCache::SetForwardInput` in [src/layer_cache.cpp](src/layer_cache.cpp), line 13.
+ *Use of smart pointers*: 
  For example in the `Layer` class, see [src/layer.h](src/layer.h), line 50 and 53.


[^1]: This terminology might be unconvential but comes from `Sequential` models of `Keras`. 
[^2]: The nice thing about Udacity Nanodegrees is that the final (Capstone) projects are entirely up to the student. 
[^3]: There might be an issue with vanishing gradients of the softmax activation function. Or even some tricky mistake in the implementation of the backpropagation algorithm.
[^4]: See [main.cpp](main.cpp) for a simple function which does this splitting in the special case of MNIST.
