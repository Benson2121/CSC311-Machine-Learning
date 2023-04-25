# CSC311 - Assignment3

## 1. Backpropagation.
In this question, you will derive the backprop updates for a particular neural net architecture. The network has one linear hidden layer. However, there are two architectural differences:
Your studies indicate that 20% of the emails are spam, i.e. Pr(t = Spam) = 0.2.

- (a) Draw the computation graph relating x, z, η, s, h, L, and the model parameters.

- (b) Derative the derivative of sigmoid function.

- (c) Derive the backprop formulas to compute the error signals for all of the model parameters, as well as x and η (recall from lecture that these are the derivatives of the cost function with respect to the variables in question). Also include the backprop formulas for all intermediate quantities needed as part of the computation.

## 2. Fitting a Naive Bayes Model
In this question, we’ll fit a Naive Bayes model to the MNIST digits using maximum likeli- hood. In addition to the mathematical derivations, you will complete the implementation in naive_bayes.py. You will fit the parameters θ and π using MLE and MAP techniques. In both cases, your fitting procedure can be written as a few simple matrix multiplication operations.

- (a) First, derive the maximum likelihood estimator (MLE) for the class-conditional pixel probabilities θ and the prior π. Derivations should be rigorous.

- (b) Derive the log-likelihood log p(t|x, θ, π) for a single training image.

- (c) Fit the parameters θ and π using the training set with MLE, and try to report the average log-likelihood per data point.

- (d) Plot the MLE estimator θ as 10 separate greyscale images, one for each class.

- (e) Derive the Maximum A posteriori Probability (MAP) estimator for the class- conditional pixel probabilities θ, using a Beta(α,β) prior

- (f) Fit the parameters θ and π using the training set with MAP estimators from the previous part, and report both the average log-likelihood per data point and the accuracy on both the training and test set.

- (g) Plot the MAP estimator θ as 10 separate greyscale images, one for each class.

- (h) List one advantage of the Naive Bayes approach and one reason why it may not be reasonable for this problem.

## 3. Logistic Regression with Gaussian Prior.

Consider a binary classification problem where the output y ∈ {0,1} and the input x.

- (a) Maximum Likelihood Estimation (MLE). Given a dataset D, write down the expression for the log-likelihood of the parameter vector θ for logistic regression. How would you optimize the resulting log-likelihood?

- (b) Derive the MAP log-likelihood of the parameter vector θ for logistic regression with the Gaussian prior.

## 4. Gaussian Discriminant Analysis.

For this question you will build classifiers to label images of handwritten digits. Each image is 8 by 8 pixels and is represented as a vector of dimension 64 by listing all the pixel values in raster scan order. The images are grayscale and the pixel values are between 0 and 1. The labels y are 0, 1, 2, . . . , 9 corresponding to which character was written in the image. There are 700 training cases and 400 test cases for each digit; they can be found in the data directory in the starter code. Using maximum likelihood, fit a set of 10 class-conditional Gaussians with a separate, full covariance matrix for each class.

- (a) Using the parameters you fit on the training set and Bayes rule, compute the average conditional log-likelihood on both the train and test set and report it.

- (b) Select the most likely posterior class for each training and test data point as your prediction, and report your accuracy on the train and test set.

- (c) Redo 4.a and 4.b by assuming that the covariance matrix is diagonal. Compare your answers and provide qualitative explanations.
The performance is worse compared with full-covariance matrix (lower likelihood and accuracy). Diagonal covariance matrix cannot model dependence between pixels.
