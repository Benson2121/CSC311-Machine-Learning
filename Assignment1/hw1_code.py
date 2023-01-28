import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import math


# Question 2 Decision Trees

# (a)

def load_data(fake_f, real_f):
    np.random.seed(21)  # Set Seed

    # Initialize lists
    x_dset = []
    y_dset = []

    # Read fake news file
    fake = open(fake_f, 'r')
    for line in fake:
        x_dset.append(line)
        y_dset.append('fake')
    fake.close()

    # Read real news file
    real = open(real_f, 'r')
    for line in real:
        x_dset.append(line)
        y_dset.append('real')
    real.close()

    # Vectorization
    vectorizer = CountVectorizer()
    x_dset = vectorizer.fit_transform(x_dset)

    # Split Datasets
    x_train, x_left, y_train, y_left = train_test_split(x_dset, y_dset, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(x_left, y_left, test_size=0.5)
    return {'x_train': x_train, 'x_val': x_val, 'x_test': x_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'feature_name': vectorizer.get_feature_names_out()}


# (b)

def get_accuracy(prediction: list, actual: list) -> float:
    correct = 0
    for i in range(len(prediction)):

        # Check Equivalence
        if prediction[i] == actual[i]:
            correct += 1

    return round(correct / len(prediction), 5)


def select_model(data: dict):

    # Initialize Hyper-parameters
    max_depths = [2 ** i for i in range(1, 6)]
    criterions = ['gini', 'entropy', 'log_loss']
    record = [[], [], []]

    # Training and Record Accuracy
    for max_depth in max_depths:
        i = 0
        for criterion in criterions:
            new_model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
            new_model.fit(data['x_train'], data['y_train'])
            val_accuracy = get_accuracy(new_model.predict(data['x_val']), data['y_val'])

            record[i].append(val_accuracy)
            print(
                f"The validation accuracy for model max_depth {max_depth} and criterion {criterion} is {val_accuracy}")
            i = i + 1

    # Select the model with the highest accuracy
    highest_accuracy = max(max(record[i]) for i in range(0, 3))
    highest_model = None
    for i in range(0, 3):
        for j in range(0, 5):
            if record[i][j] == highest_accuracy:
                highest_model = (max_depths[j], criterions[i])
                break
        if highest_model is not None:
            break
    print(
        f"""The model that reaches the highest accuracy has max_depth {highest_model[0]} and criterion {highest_model[1]}
            The corresponding accuracy is {highest_accuracy}""")

    # Find the best model and get test_accuracy
    final_model = DecisionTreeClassifier(max_depth=highest_model[0], criterion=highest_model[1])
    final_model.fit(data['x_train'], data['y_train'])
    test_accuracy = get_accuracy(final_model.predict(data['x_test']), data['y_test'])
    print(f"Our best model has test accuracy {test_accuracy}")

    # Plot
    plt.plot(max_depths, record[0], c='r', label='gini')
    plt.scatter(max_depths, record[0], c='r', label='gini')
    plt.plot(max_depths, record[1], c='g', label='entropy')
    plt.scatter(max_depths, record[1], c='g', label='entropy')
    plt.plot(max_depths, record[2], c='b', label='log_loss')
    plt.scatter(max_depths, record[2], c='b', label='log_loss')
    plt.title('val accuracy vs. max_depth')
    plt.xlabel('max_depths')
    plt.ylabel('val accuracy')
    plt.legend()
    plt.show()

    return final_model


# (c)(d)

def entropy(events_num: list) -> float:
    all_en = []
    for event_num in events_num:
        event_prob = event_num / (sum(events_num))
        single = event_prob * math.log(event_prob, 2)
        all_en.append(single)
    return -sum(all_en)


def compute_information_gain(left: list, right: list) -> float:
    # Compute H(Y)
    h_y = entropy([left[0] + right[0], left[1] + right[1]])

    # Compute H(Y | X)
    h_left = entropy(left)
    left_prob = sum(left) / sum(left + right)
    h_right = entropy(right)
    right_prob = sum(right) / sum(left + right)
    h_y_under_x = left_prob * h_left + right_prob * h_right

    return round(h_y - h_y_under_x, 5)


def get_split_information_gain(x_data, y_data, word_index, decision_bound: float) -> float:
    x_data_array = x_data.toarray()
    left_fake, left_real, right_fake, right_real = 0, 0, 0, 0
    for i in range(len(x_data_array)):
        dp = x_data_array[i]
        label = y_data[i]
        decision = (dp[word_index] <= decision_bound)
        if decision:
            if label == 'fake':
                left_fake += 1
            else:
                left_real += 1
        else:
            if label == 'fake':
                right_fake += 1
            else:
                right_real += 1
    return compute_information_gain([left_fake, left_real], [right_fake, right_real])


if __name__ == '__main__':
    data = load_data('clean_fake.txt', 'clean_real.txt')
    model = select_model(data)
    features = data['feature_name']
    fig = plt.figure(figsize=(20, 10))
    tree.plot_tree(model, filled=True, max_depth=2, class_names=model.classes_, feature_names=features)
    print(f"Information Gain for the topmost split: {compute_information_gain([649, 1275], [251, 111])}")
    print("===" * 25)
    print("===" * 25)
    print("Some other words: ")
    words = ["trump", "and", "2016", "fake", "news"]
    for word in words:
        word_index = np.where(features == word)
        print(f"Information Gain for the split based on word {word} is "
              f"{get_split_information_gain(data['x_train'], data['y_train'], word_index, 0.5)}")

