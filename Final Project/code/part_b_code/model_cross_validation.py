from sklearn.model_selection import KFold

from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import *


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        x = self.g(inputs)
        x = F.sigmoid(x)
        x = self.h(x)
        out = F.sigmoid(x)
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: (lst_train_loss, lst_val_acc)
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Record the training loss and validation accuracy.
    lst_train_loss = []
    lst_val_acc = []

    # Define optimizers and loss function.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            # Add the regularizer.
            loss = torch.sum((output - target) ** 2.) + lamb / 2 * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        # Evaluate the model on the validation set.
        valid_acc = evaluate(model, zero_train_data, valid_data)
        lst_train_loss.append(train_loss)
        lst_val_acc.append(valid_acc)

        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch + 1, train_loss, valid_acc))
    return lst_train_loss, lst_val_acc


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def re_sampling(data, size) -> torch.Tensor:
    """
    Bootstrapping the training set.
    :param data: A Tensor
    :return: A Tensor
    """
    resample = torch.Tensor(size, 3)
    i = 0
    while i < size:
        index = np.random.randint(0, len(data))
        resample[i] = data[index]
        i += 1
    return resample


def cross_validation(model, data, origin_valid, matrix_size, lr, lamb, num_epoch, n_splits):
    """ This function performs k-fold cross validation on the training data.

    :param model: Module
    :param data: 2D FloatTensor
    :param origin_valid: A dictionary
    :param matrix_size: tuple
    :param lr: float
    :param lamb: float
    :param num_epoch: int
    :param n_splits: int
    :return
    """
    # Define the K-fold Cross Validator
    kf = KFold(n_splits=n_splits)

    accuracies = []
    size = len(data)

    iteration = 0
    for train_index, valid_index in kf.split(data):
        # Split the data into training and testing.
        train_data, valid_data = data[train_index], data[valid_index]

        # Resampling the training set.
        train_data = re_sampling(train_data, size)

        # Convert the training data to matrix.
        train_matrix = np.empty(matrix_size)
        train_matrix[:] = np.nan

        for i in range(len(train_data)):
            train_matrix[int(train_data[i][0])][int(train_data[i][1])] = train_data[i][2]

        # Replace the nan with 0.
        zero_train_matrix = np.nan_to_num(train_matrix)

        # Convert the data to FloatTensor.
        zero_train_matrix = torch.FloatTensor(zero_train_matrix)
        train_matrix = torch.FloatTensor(train_matrix)

        # Train the model.
        print("Iteration: ", iteration + 1)
        _, val_acc = train(model, lr, lamb, train_matrix, zero_train_matrix, origin_valid, num_epoch)

        # Evaluate the model on the testing set.
        origin_accuracy = evaluate(model, zero_train_matrix, origin_valid)
        accuracies.extend(val_acc)

        # Print the accuracy.
        print("The validation accuracy of the model on the validation set is: ", origin_accuracy)

        iteration += 1

    return accuracies


def main():
    # Load the data.
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    matrix_size = zero_train_matrix.shape

    # Convert the data to (num_student * num_question, 3) tensor.
    transfer_data = torch.Tensor(train_matrix.shape[0] * train_matrix.shape[1], 3)
    index = 0
    for i in range(train_matrix.shape[0]):
        for j in range(train_matrix.shape[1]):
            if not np.isnan(train_matrix[i][j]):
                transfer_data[index][0] = i
                transfer_data[index][1] = j
                transfer_data[index][2] = train_matrix[i][j]
                index += 1

    transfer_data = transfer_data[torch.randperm(transfer_data.size()[0])]
    # Tune model hyperparameters.
    k_set = [10, 50, 100, 200, 500]
    splits = [3, 5, 10]
    lamb_set = [0, 0.001, 0.01, 0.1, 0.3, 1]
    lr_set = [0.001, 0.005, 0.01, 0.05, 0.1, 1]
    num_epoch_set = [3, 5, 7, 10, 20, 30, 50]
    model = AutoEncoder(train_matrix.shape[1], k=k_set[3])
    accuracies = cross_validation(model, transfer_data, valid_data, matrix_size, lr_set[2], lamb_set[1],
                                  num_epoch_set[1], splits[1])

    iters = range(len(accuracies))
    plt.plot(iters, accuracies, 'b', label='valid accuracy')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    for i in range(0, len(accuracies), splits[1]):
        plt.axvline(x=i, color='r', linestyle='--')
    plt.title("valid accuracy")
    plt.legend(loc="upper right")
    plt.show()

    print("The final validation accuracy of the model is: ", evaluate(model, zero_train_matrix, valid_data))
    print("The testing accuracy of the model is: ", evaluate(model, zero_train_matrix, test_data))


if __name__ == "__main__":
    main()
