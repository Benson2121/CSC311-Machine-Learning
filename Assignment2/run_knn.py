from sklearn.metrics import confusion_matrix

from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = np.int_(np.mean(valid_labels, axis=1) >= 0.5)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def get_accuracy(C):
    """ Compute accuracy given Numpy array confusion matrix C. Returns a floating point value """
    return round(np.sum(np.diag(C)) / np.sum(C), 4)


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    k_list = [1, 3, 5, 7, 9]
    accuracies = []
    for k in k_list:
        valid_labels = knn(k, train_inputs, train_targets, valid_inputs)
        C = confusion_matrix(valid_targets, valid_labels)
        accuracies.append(get_accuracy(C))

    # Plot the accuracies
    plt.scatter(k_list, accuracies)
    plt.plot(k_list, accuracies)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('k vs. accuracy')
    plt.show()

    print("I choose k = 5, since it's the median value for highest accuracies")
    k = 5
    valid_labels = knn(k, train_inputs, train_targets, valid_inputs)
    C = confusion_matrix(valid_targets, valid_labels)
    print(f'Accuracy on validation set: {get_accuracy(C)}')
    test_labels = knn(k, train_inputs, train_targets, test_inputs)
    C = confusion_matrix(test_targets, test_labels)
    print(f'Accuracy on test set: {get_accuracy(C)}')

    print("===" * 20)
    print("===" * 20)
    print('Try k = 7 (5 + 2)')
    valid_labels = knn(k, train_inputs, train_targets, valid_inputs)
    C = confusion_matrix(valid_targets, valid_labels)
    print(f'Accuracy on validation set: {get_accuracy(C)}')
    test_labels = knn(k, train_inputs, train_targets, test_inputs)
    C = confusion_matrix(test_targets, test_labels)
    print(f'Accuracy on test set: {get_accuracy(C)}')

    print("===" * 20)
    print("===" * 20)
    print('Try k = 3 (5 - 2)')
    valid_labels = knn(k, train_inputs, train_targets, valid_inputs)
    C = confusion_matrix(valid_targets, valid_labels)
    print(f'Accuracy on validation set: {get_accuracy(C)}')
    test_labels = knn(k, train_inputs, train_targets, test_inputs)
    C = confusion_matrix(test_targets, test_labels)
    print(f'Accuracy on test set: {get_accuracy(C)}')

    print("===" * 20)
    print("===" * 20)
    print("As we change the value of k, we can see that the accuracy don' have a big change in this question.")
    print("However, we expect the accuracy to decrease when we change the value of k.")
    print("When k is smaller than 5, the model would more likely to underfit the data.")
    print("When k is larger than 5, the model would more likely to overfit the data.")


if __name__ == "__main__":
    run_knn()
