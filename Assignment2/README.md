# CSC311 - Assignment2

Please refer to the handout.pdf for the full assignment.

## 1. Nearest Neighbours and the Curse of Dimensionality. 

In this question, you will verify the claim from lecture that “most” points in a high-dimensional space are far away from each other, and also approximately the same distance. There is a very neat proof of this fact which uses the properties of expectation and variance.

(a) Suppose we have a classification dataset where each data point has one feature. The feature takes on a real value between [0, 1]. What is the minimum number of data points we need to guarantee that any new test point is within (≤) 0.01 of an old point?

(b) Explain why such a guarantee is more difficult to maintain when we are working on a problem with 10 features.

(c) For each choice of dimension d ∈ [20, 21, 22, ..., 210], sample 100 points from the unit cube, and record the following average distances between all pairs of points, as well as the standard deviation of the distances.

(d) In this question, we aim to verify our simulations in part (c) by deriving the analytical form of averaged Euclidean distance and variance of Euclidean distance.

(e) In probability theory, one can derive that P(|Z − E[Z]| ≥ a) ≤ Var[Z]/a^2 for any random variable Z. (This fact is known as Markov’s Inequality.) Based on your answer to part (d), explain why does this support the claim that in high dimensions, “most points are approximately the same distance”?

## 2. Decision Trees.
In this question, you will use the scikit-learn decision tree classifier to classify real vs. fake news headlines. The aim of this question is for you to read the scikit-learn API and get comfortable with training/validation splits.

We will use a dataset of 1298 “fake news” headlines (which mostly include headlines of articles classified as biased, etc.) and 1968 “real” news headlines, where the “fake news” headlines are from [fake-news-data](https://www.kaggle.com/mrisdal/fake-news/data) and “real news” headlines are from [real-news-data](https://www.kaggle.com/therohk/million-headlines). The data were cleaned by removing words from fake news titles that are not a part of the headline, removing special characters from the headlines, and restricting real news headlines to those after October 2016 containing the word “trump”.

(a) Write a function load_data which loads the data, preprocesses it using a [vectorizer](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text), and splits the entire dataset randomly into 70% training, 15% vali- dation, and 15% test examples.

(b) Write a function select_model which trains the decision tree classifier using at least 5 different values of max_depth, as well as three different split criteria (infor- mation gain, log loss and Gini coefficient), evaluates the performance of each one on the validation set, and prints the resulting accuracies of each model.

(c) Now let’s stick with the hyperparameters which achieved the highest validation accuracy. Extract and visualize the first two layers of the tree.

## 3. Regularized Linear Regression.
In lecture, we saw that regression models with too much capacity can overfit the training data and fail to generalize. We also saw that one way to improve generalization is regularization: adding a term to the cost function which favors some explanations over others.

(a) Determine the gradient descent update rules for the regularized cost function. You may notice that the absolute value function is not differentiable everywhere,
in particular at 0.

(b) For the remaining part of the question, consider the special case where λ1 = 0. In other words, we only apply the l2 penalty. It is possible to solve this regularized regression problem, also called Ridge Regression, directly by setting the partial derivatives equal to zero.

(c) Based on your answer to part (b), determine formulas for A and c, and derive a closed-form solution for the parameter w. Note that, as usual, the inputs are organized into a design matrix X with one row per training example.
