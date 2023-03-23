'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        means[i] = np.mean(train_data[train_labels == i], axis=0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    means = compute_mean_mles(train_data, train_labels)
    covariances = np.zeros((10, 64, 64))
    coef = 0.01
    # Compute covariances
    for i in range(10):
        x_u = train_data[train_labels == i] - means[i]
        covariances[i] = (x_u.T @ x_u) / np.sum(train_labels == i) + np.eye(64) * coef
    return covariances

# def compute_sigma_mles(train_data, train_labels):
#     '''
#     Compute the diagonal covariance estimate for each digit class.
#     This function is for question 4(c).
#
#     Should return a three dimensional numpy array of shape (10, 64, 64)
#     consisting of a diagonal covariance matrix for each digit class.
#     '''
#     means = compute_mean_mles(train_data, train_labels)
#     covariances = np.zeros((10, 64, 64))
#     coef = 0.01
#     # Compute diagonal covariances
#     for i in range(10):
#         x_u = train_data[train_labels == i] - means[i]
#         covariances[i] = np.diag(np.var(x_u, axis=0)) + np.eye(64) * coef
#     return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    d = 64
    n = digits.shape[0]
    likelihood = np.zeros((n, 10))
    for i in range(10):
        x_u = digits - means[i]
        const = -0.5 * d * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(covariances[i]))
        likelihood[:, i] = const - 0.5 * np.sum(x_u @ np.linalg.inv(covariances[i]) * x_u, axis=1)
    return likelihood



def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    likelihood_p_x_given_y = generative_likelihood(digits, means, covariances)
    p_y = np.array([0.1] * 10)
    con_likelihood = likelihood_p_x_given_y + np.log(p_y) - np.log(np.sum(np.exp(likelihood_p_x_given_y), axis=1, keepdims=True))
    return con_likelihood

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    one_hot_labels = np.eye(10)[labels.astype(int)]
    return np.mean(np.sum(cond_likelihood * one_hot_labels, axis=1))

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    return np.argmax(cond_likelihood, axis=1)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    train_avg_ll = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg_ll = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print('Average Train LL: {}'.format(train_avg_ll))
    print('Average Test LL: {}'.format(test_avg_ll))
    print("Train Accuracy: ", np.mean(classify_data(train_data, means, covariances) == train_labels))
    print("Test Accuracy: ", np.mean(classify_data(test_data, means, covariances) == test_labels))

if __name__ == '__main__':
    main()
