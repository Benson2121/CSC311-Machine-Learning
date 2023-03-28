from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    k_list = [1, 6, 11, 16, 21, 26]
    print("=== User-based ===")
    user_based_accuracies = []
    for k in k_list:
        val_accuracy = knn_impute_by_user(sparse_matrix, val_data, k)
        print("k = {}, Validation Accuracy = {}".format(k, val_accuracy))
        user_based_accuracies.append(val_accuracy)
    plt.scatter(k_list, user_based_accuracies)
    plt.plot(k_list, user_based_accuracies)
    plt.xlabel('k')
    plt.ylabel('user_based_accuracy')
    plt.title('k vs. user_based_accuracy')
    plt.show()
    k_user_star = k_list[user_based_accuracies.index(max(user_based_accuracies))]

    print(f"I choose k* = {k_user_star}, with the validation accuracy of {max(user_based_accuracies)}")
    print("Test accuracy: {}\n".format(knn_impute_by_user(sparse_matrix, test_data, k_user_star)))

    print("=== Item-based ===")
    item_based_accuracies = []
    for k in k_list:
        val_accuracy = knn_impute_by_item(sparse_matrix, val_data, k)
        print("k = {}, Validation Accuracy = {}".format(k, val_accuracy))
        item_based_accuracies.append(val_accuracy)
    plt.scatter(k_list, item_based_accuracies)
    plt.plot(k_list, item_based_accuracies)
    plt.xlabel('k')
    plt.ylabel('item_based_accuracy')
    plt.title('k vs. item_based_accuracy')
    plt.show()
    k_item_star = k_list[item_based_accuracies.index(max(item_based_accuracies))]

    print(f"I choose k* = {k_item_star}, with the validation accuracy of {max(item_based_accuracies)}")
    print("Test accuracy: {}\n".format(knn_impute_by_item(sparse_matrix, test_data, k_item_star)))




    #####################################################################


if __name__ == "__main__":
    main()
