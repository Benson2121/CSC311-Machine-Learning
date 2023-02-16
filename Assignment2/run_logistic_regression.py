from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    hyperparameters = {
        "learning_rate": 0.01,
        "weight_regularization": 0,
        "num_iterations": 1000
    }
    weights = 0.001 * np.random.randn(M + 1, 1)

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    num_iterations = range(hyperparameters["num_iterations"])
    train_loss_data = []
    train_accuracy_data = []
    valid_loss_data = []
    valid_accuracy_data = []

    for i in range(hyperparameters["num_iterations"]):
        train_f, df, train_prediction = logistic(weights, train_inputs, train_targets, hyperparameters)

        train_loss, train_accuracy = evaluate(train_targets, train_prediction)
        valid_y = logistic_predict(weights, valid_inputs)
        valid_loss, valid_accuracy = evaluate(valid_targets, valid_y)

        train_loss_data.append(train_loss)
        train_accuracy_data.append(train_accuracy)
        valid_loss_data.append(valid_loss)
        valid_accuracy_data.append(valid_accuracy)

        weights = weights - hyperparameters["learning_rate"] * df

        if i == 999:
            print(f"Final train cross entropy loss: {train_loss:.4f}")
            print(f"Final train classification accuracy: {train_accuracy:.4f}")
            print(f"Final validation cross entropy loss: {valid_loss:.4f}")
            print(f"Final validation classification accuracy: {valid_accuracy:.4f}")


    # Plot learning curves
    plt.plot(num_iterations, train_loss_data, label="train_loss")
    plt.plot(num_iterations, valid_loss_data, label="valid_loss")
    plt.xlabel("Iteration")
    plt.legend()
    plt.title("Cross Entropy Loss")
    plt.show()

    plt.plot(num_iterations, train_accuracy_data, label="train_accuracy")
    plt.plot(num_iterations, valid_accuracy_data, label="valid_accuracy")
    plt.xlabel("Iteration")
    plt.legend()
    plt.title("Classification Accuracy")
    plt.show()

    # Compute test error
    test_y = logistic_predict(weights, test_inputs)
    test_ce, test_acc = evaluate(test_targets, test_y)
    print(f"Test cross entropy loss: {test_ce:.4f}")
    print(f"Test classification accuracy: {test_acc:.4f}")


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("finite difference =", diff)


if __name__ == "__main__":
    run_logistic_regression()