# Some development notes

# TODOs 
  These are divided into two categories 
  #A: important and possibly implemented before submission
  #B: not so urgent (possiblity still added before submission)

# Design issue with Perceptrons and Layers
  Concerning backpropagation it makes more sense to build perceptrons (affine linear transformation
  with activation function) from various layers. 
  The point is: we need to keep the results of both the affine linear transformation and after applying the 
  (vectorized) activation function.

# Matrix class?
  It would have been better to implement a Matrix class which deals with transpose, evaluating etc.
  But this would have required another refactoring and is therefore postponed.
  In fact, in the future we would prefer to use external libraries (e.g. Blaze or Eigen) for such 
  computations.

# LayerCache
  Needed for connecting layers.

# TODOs
  + Where to use move semantics or passing by reference? 
  + Use _transformation in Layer::Forward.
  + Factory pattern to create Layers?
  + Do we really need pointers to Transformations in Layer?

# Ideas for the future (which might never be implemented...)
  + Provide options for other optimizers, not just stochastic gradient descent.
    This would have several consequences:
    + Include a 'backward transformation' in the Layer class. 
    + It might be useful to split LayerCache into forward and backward cache.
      Note that on an abstract level (at least for sequential NNs), forward and backward 
      pass work the same. The only tricky part is to correctly initialize the backward
      transformation.
