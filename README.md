# Sequential neural networks in modern `C++` (almost) from scratch

The aim of this project is to implement a basic API in modern `C++` to build and train sequential neural networks[^1], i.e. neural networks whose layers have exactly one input and one output layer. It is still work in progress and will be my Capstone project[^2] in the [Udacity C++ Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213).

## Why in `C++` and not e.g. `Python`?
After having learnt the basics of Neural networks (NNs), my naive thought as a pure Mathematician (at least back then) was: "Ok, that's just the chain rule." However, the blessing and curse of a Mathematician is that you want to understand things on a foundational level. So I asked myself how NNs are actually implemented. Implementing them in Python together with NumPy does a lot of things in the background: deciding between pass by value vs. pass by reference, using precompiled `C`-code etc. Therefore I decided to implement NNs in `C++` with as little help as reasonable[^3] . It turns out that this is a great opportunity to learn techniques from modern `C++`, e.g. smart pointers, in more detail and to apply them.


[^1]: This terminology might be unconvential but comes from `Sequential` models of `Keras`. 
[^2]: The nice thing about Udacity Nanodegrees is that the final (Capstone) projects are entirely up to the student. 
[^3]: Implementing a fully functioning Matrix class as well would be too much for a Capstone project. So we use `Eigen` instead.
