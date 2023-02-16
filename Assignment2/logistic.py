from utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    ones = np.ones((data.shape[0], 1))
    data_with_bias = np.hstack((data, ones))
    logits = data_with_bias @ weights
    y = sigmoid(logits)

    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """

    ce = -np.mean(targets * np.log(y) + (1 - targets) * np.log(1 - y))
    frac_correct = np.mean(targets == np.round(y))

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)
    f = evaluate(targets, y)[0]
    f += hyperparameters['weight_regularization'] * np.sum(weights ** 2)

    ones = np.ones((data.shape[0], 1))
    data_with_bias = np.hstack((data, ones))

    grad_ce = np.transpose(data_with_bias) @ (y - targets) / data_with_bias.shape[0]
    grad_reg = 2 * hyperparameters['weight_regularization'] * weights
    df = grad_ce + grad_reg
    return f, df, y
