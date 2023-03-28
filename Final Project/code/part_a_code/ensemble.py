from utils import *
from item_response import *
import matplotlib.pyplot as plt
import numpy as np
def re_sampling(data) -> dict:
    """
    Bootstrapping the training set.
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
    """
    resample = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    i = 0
    while i < len(data['user_id']):
        index = np.random.randint(0, len(data['user_id']))
        resample['user_id'].append(data['user_id'][index])
        resample['question_id'].append(data['question_id'][index])
        resample['is_correct'].append(data['is_correct'][index])
        i += 1
    return resample

def evaluate_ensemble(data, theta_list, beta_list):
    """
    Evaluate the accuracy of the model.
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param theta_list: A list of theta
    :param beta_list: A list of beta
    :return: float
    """
    final_pred = []
    for i, q in enumerate(data["question_id"]):
        pred = []
        u = data["user_id"][i]
        for k in range(len(theta_list)):
            u = data["user_id"][i]
            x = (theta_list[k][u] - beta_list[k][q]).sum()
            p_a = sigmoid(x)
            pred.append(p_a)
        final_pred.append(np.mean(pred) >= 0.5)
    final_pred = np.array(final_pred)
    return np.sum((data["is_correct"] == final_pred)) / len(data["is_correct"])


def ensemble_method_1(data, val_data, lr, iterations, num_models):
    """
    Train IRT model with ensemble method 1.
    (We train multiple IRT models and use the majority vote to predict the answer.)
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param lr: float
    :param iterations: int
    :param num_models: int
    :return: (theta_list, beta_list)
    """
    theta_list = []
    beta_list = []

    for i in range(num_models):
        print(f"-------------Training IRT for model {i + 1} start--------------")
        data = re_sampling(data)
        theta, beta, train_acc_lst, val_acc_lst, train_lld_lst, val_lld_lst = irt(data, val_data, lr, iterations)
        theta_list.append(theta)
        beta_list.append(beta)
        print(f"-------------Training IRT for model {i + 1} complete-----------\n")
    return theta_list, beta_list

# def ensemble_method_2(data, val_data, lr, iterations, num_models):
#     """
#     Train IRT model with ensemble method 2.
#     (When calculating the gradient, we use the average of the gradients from all the models)
#     :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
#     :param val_data: A dictionary {user_id: list, question_id: list, is_correct: list}
#     :param lr: float
#     :param iterations: int
#     :param num_models: int
#     :return: (theta, beta, train_acc_lst, val_acc_lst, train_lld_lst, val_lld_lst)
#     """
#     theta = np.random.normal(0, 0.1, (len(data["user_id"]), 1))
#     beta = np.random.normal(0, 0.1, (len(data["question_id"]), 1))
#
#     train_acc_lst = []
#     val_acc_lst = []
#     train_lld_lst = []
#     val_lld_lst = []
#
#     for i in range(iterations):
#         train_acc_lst.append(evaluate(data, theta, beta))
#         val_accuracy = evaluate(val_data, theta, beta)
#         val_acc_lst.append(val_accuracy)
#         neg_lld = neg_log_likelihood(data, theta, beta)
#         train_lld_lst.append(-neg_lld)
#         val_lld_lst.append(-neg_log_likelihood(val_data, theta, beta))
#         print("Iteration: {} \t Training NLLK: {} \t Validation Accuracy: {}".format(i + 1, neg_lld, val_accuracy))
#         for j in range(num_models):
#             re_sample = re_sampling(data)
#             theta, beta = update_theta_beta(re_sample, lr, theta, beta)
#     return theta, beta, train_acc_lst, val_acc_lst, train_lld_lst, val_lld_lst

def main():

    data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    lr = 0.005
    iterations = 10
    num_models = 3
    iteration_range = range(1, iterations + 1)

    # Ensemble method 1
    theta_list, beta_list = ensemble_method_1(data, val_data, lr, iterations, num_models)
    val_acc = evaluate_ensemble(val_data, theta_list, beta_list)
    test_acc = evaluate_ensemble(test_data, theta_list, beta_list)
    print(f"Final Validation accuracy: {val_acc}")
    print(f"Test accuracy: {test_acc}")

    # Ensemble method 2
    # theta, beta, train_acc_lst, val_acc_lst, train_lld_lst, val_lld_lst = ensemble_method_2(data, val_data, lr, iterations, num_models)
    #
    # plt.scatter(iteration_range, train_acc_lst)
    # plt.scatter(iteration_range, val_acc_lst)
    # plt.plot(iteration_range, train_acc_lst)
    # plt.plot(iteration_range, val_acc_lst)
    # plt.xlabel("Iterations")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy vs. Iterations")
    # plt.legend(["Train", "Validation"])
    # plt.show()
    #
    # plt.scatter(iteration_range, train_lld_lst)
    # plt.scatter(iteration_range, val_lld_lst)
    # plt.plot(iteration_range, train_lld_lst)
    # plt.plot(iteration_range, val_lld_lst)
    # plt.xlabel("Iterations")
    # plt.ylabel("Log-Likelihood")
    # plt.title("Log-Likelihood vs. Iterations")
    # plt.legend(["Train", "Validation"])
    # plt.show()
    #
    # print("Final Validation Accuracy: {}".format(evaluate(val_data, theta, beta)))
    # print("Test Accuracy: {}".format(evaluate(test_data, theta, beta)))

if __name__ == "__main__":
    main()
