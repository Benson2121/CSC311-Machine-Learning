import matplotlib.pyplot as plt
from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    log_lklihood = 0
    for i in range(len(data['user_id'])):
        u = data['user_id'][i]
        q = data['question_id'][i]
        x = theta[u] - beta[q]
        log_lklihood += data['is_correct'][i] * x - np.log(1 + np.exp(x))
    neg_lklihood = -log_lklihood
    #####################################################################
    return float(neg_lklihood)


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    theta_update = np.zeros(theta.shape)
    beta_update = np.zeros(beta.shape)
    #####################################################################
    for i in range(len(data['user_id'])):
        u = data['user_id'][i]
        q = data['question_id'][i]
        x = theta[u] - beta[q]
        theta_update[u] += (data['is_correct'][i] - sigmoid(x))
        beta_update[q] += (sigmoid(x) - data['is_correct'][i])
    theta += lr * theta_update
    beta += lr * beta_update
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, train_acc_lst, val_acc_lst, train_lld_lst, val_lld_lst)
    """
    theta = np.random.normal(0, 0.1, (len(data["user_id"]), 1))
    beta = np.random.normal(0, 0.1, (len(data["question_id"]), 1))

    train_acc_lst = []
    val_acc_lst = []
    train_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        train_acc_lst.append(evaluate(data, theta, beta))
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_lld_lst.append(-neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_lld_lst.append(-val_neg_lld)
        print("Iteration: {} \t Training NLLK: {} \t Validation Accuracy: {}".format(i + 1, neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, train_acc_lst, val_acc_lst, train_lld_lst, val_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    lr = 0.005
    iterations = 10
    iteration_range = range(1, iterations + 1)
    theta, beta, train_acc_lst, val_acc_lst, train_lld_lst, val_lld_lst = irt(train_data, val_data, lr, iterations)

    plt.scatter(iteration_range, train_acc_lst)
    plt.scatter(iteration_range, val_acc_lst)
    plt.plot(iteration_range, train_acc_lst)
    plt.plot(iteration_range, val_acc_lst)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Iterations")
    plt.legend(["Train", "Validation"])
    plt.show()

    plt.scatter(iteration_range, train_lld_lst)
    plt.scatter(iteration_range, val_lld_lst)
    plt.plot(iteration_range, train_lld_lst)
    plt.plot(iteration_range, val_lld_lst)
    plt.xlabel("Iterations")
    plt.ylabel("Log-Likelihood")
    plt.title("Log-Likelihood vs. Iterations")
    plt.legend(["Train", "Validation"])
    plt.show()

    print("Final Validation Accuracy: {}".format(evaluate(val_data, theta, beta)))
    print("Test Accuracy: {}".format(evaluate(test_data, theta, beta)))
    #####################################################################

    #####################################################################
    # Plot the probability of correct response vs. theta for three questions
    theta_range = np.linspace(-max(theta)-0.2, max(theta)+0.2, 1000)
    # Select three questions
    j1 = 311
    j2 = beta.argmax()
    j3 = beta.argmin()

    # Compute probability of correct response for each question
    p_j1 = 1 - sigmoid(theta_range - beta[j1])
    p_j2 = 1 - sigmoid(theta_range - beta[j2])
    p_j3 = 1 - sigmoid(theta_range - beta[j3])

    # Plot the curves
    plt.plot(theta_range, p_j1, color='r', label='Question {}'.format(j1))
    plt.plot(theta_range, p_j2, color='g', label='Question {}'.format(j2))
    plt.plot(theta_range, p_j3, color='b', label='Question {}'.format(j3))
    plt.axvline(x=theta[j1], color='r', linestyle='--', label='Best Theta ({})'.format(j1))
    plt.axvline(x=theta[j2], color='g', linestyle='--', label='Best Theta ({})'.format(j2))
    plt.axvline(x=theta[j3], color='b', linestyle='--', label='Best Theta ({})'.format(j3))
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'Probability of Correct Response')
    plt.title('Probability of Correct Response vs. Theta')
    plt.legend()
    plt.show()
    #####################################################################


if __name__ == "__main__":
    main()
