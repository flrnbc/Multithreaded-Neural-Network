# Some development notes

# TODOs 
  These are divided into two categories 
  #A: important and possibly implemented before submission
  #B: not so urgent (possiblity still added before submission)

# Activation vs ActivationFct
  We (loosely) follow the rule: Activation refers to the name of an 
  activation function ActivationFct. The point we are trying to make:
  PerceptronData is 'pure data' whereas Perceptron contains the actual
  functions as well.

# PerceptronData and Perceptron
  We decided to compose Perceptron from PerceptronData. However, this 
  led to some boilerplate code since we wanted to faciliate the access
  to the perceptron data (e.g. Rows(), Cols()). So some other design
  might be more elegant.

# LayerBase vs Perceptron
  The class LayerBase simply adds additional data to a Perceptron. 
  It is not strictly needed. We mainly use it to solve the problem 
  of input and output layers (having no previous or next layer respectively).
  By implementing LayerBase as a concrete class instead of an abstract class,
  we avoid overwriting the setters & getters for both InputLayer and OutputLayer  in the same way. 
  Another solution would be to include a 'NULL' layer but found the above more
  convenient to implement. 

# Design issue with Perceptrons and Layers
  Concerning backpropagation it makes more sense to build perceptrons from various layers. The point
  is: we need to keep the results of both the affine linear transformation and after applying the 
  (vectorized) activation function.

# Matrix class?
  It would have been better to implement a Matrix class which deals with transpose, evaluating etc.
  But this would have required another refactoring and is therefore postponed.


# TODOs
  + Where to use move semantics or passing by reference? 
  + Implement Forward() in Transformation class
  + Factory pattern to create Layers?
