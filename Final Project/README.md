# CSC311 - Final Project

## 1. Introduction
One of CSC311’s main objectives is to prepare you to apply machine learning algorithms to real-world tasks. The final project aims to help you get started in this direction. You will be performing the following tasks:

- Try out existing algorithms to real-world tasks.

- Modify an existing algorithm to improve performance.

- Write a short report analyzing the result.

The final project is not intended to be a stressful experience. It is a good chance for you to experiment, think, play, and hopefully have fun. These tasks are similar to what you may be doing daily as a data analyst/scientist or machine learning engineer.

We have an existing project you can trackle and also a second open-ended project option described in Section 6. Both options should have roughly equal amounts of work. We will award a prize to the best projects, determined by the instructional team.

## 2. Background & Task


Online education services, such as Khan Academy and Coursera, provide a broader audience with access to high-quality education. On these platforms, students can learn new materials by watching a lecture, reading course material, and talking to instructors in a forum. However, one disadvantage of the online platform is that it is challenging to measure students’ understanding of the course material.

To deal with this issue, many online education platforms include an assessment component to ensure that students understand the core topics. The assessment component is often composed of diagnostic questions, each a multiple choice question with one correct answer. The diagnostic question is designed so that each of the incorrect answers highlights a common misconception. When students incorrectly answer the diagnostic question, it reveals the nature of their misconception and, by understanding these misconceptions, the platform can offer additional guidance to help resolve them.

In this project, you will build machine learning algorithms to predict whether a student can correctly answer a specific diagnostic question based on the student’s previous answers to other questions and other students’ responses. Predicting the correctness of students’ answers to as yet unseen diagnostic questions helps estimate the student’s ability level in a personalized education platform. Moreover, these predictions form the groundwork for many advanced customized tasks. For instance, using the predicted correctness, the online platform can automatically recommend a set of diagnostic questions of appropriate difficulty that fit the student’s background and learning status.


You will begin by applying existing machine learning algorithms you learned in this course. You will then compare the performances of different algorithms and analyze their advantages and disadvantages. Next, you will modify existing algorithms to predict students’ answers with higher accuracy. Lastly, you will experiment with your modification and write up a short report with the results.

## 3. Data

We subsampled answers of 542 students to 1774 diagnostic questions from the dataset provided by Eedi1, an online education platform that is currently being used in many schools.
The data is provided in the data directory in the starter code. 

### 3.1 Primary Data

The primary data, train_data.csv, is the main dataset you will be using to train the learning algorithms throughout the project. There is also a validation set valid_data.csv that you should use for model selection and a test set test_data.csv that you should use for reporting the final performance. All primary data csv files are composed of 3 columns:

- **question id**: ID of the question answered (starts from 0).
- **user id**: ID of the student who answered the question (starts from 0).
- **is correct**: Binary indicator whether the student’s answer was correct (0 is incorrect, 1 is correct).


We also provide a sparse matrix, sparse_matrix.npz, where each row corresponds to the user id and each column corresponds to the question id. An illustration of the sparse matrix is shown in figure 2. The correct answer given a pair of (user id, question id) will have an entry 1 and an incorrect answer will have an entry 0. Answers with no observation and held-out data (that will be used for validation and test) will have an entry NaN (np.NaN).

### 3.2 Question Metadata

We also provide the question metadata, question_meta.csv, which contains the following columns:

- **question_id**: the question ID.
- **subject id**: The subject of the question covered in an area of mathematics. The text de- scription of each subject is provided in subject_meta.csv.

### 3.3 Student Metadata

Lastly, we provide the student metadata, student_meta.csv, that is composed of the following columns:

- **user id**: ID of the student who answered the question (starts from 0).
- **gender**: Gender of the student, when available. 1 indicates a female, 2 indicates a male, and 0 indicates unspecified.
- **data of birth**: Birth date of the student, when available.
- **premium pupil**: Student’s eligibility for free school meals or pupil premium due to being financially disadvantaged, when available.

## 4. Part A: Applying Existing Algorithms

In the first part of the project, you will implement and apply various machine learning algorithms you studied in the course to predict students’ correctness of a given diagnostic question. Review the course notes if you don’t recall the details of each algorithm. For this part, you will only be using the primary data: train_data.csv, sparse_matrix.npz, valid_data.csv, and test_data.csv. Moreover, you may use the helper functions provided in utils.py to load the dataset and evaluate your model. You may also use any functions from packages NumPy, Scipy, Pandas, and PyTorch. Make sure you understand the code instead of using it as a black box.

### 4.1 k-Nearest Neighbor.

In this problem, using the provided code at part_a/knn.py, you will experiment with k-Nearest Neighbor (kNN) algorithm.

The provided kNN code performs collaborative filtering that uses other students’ answers to predict whether the specific student can correctly answer some diagnostic questions. In particular, the starter code implements user-based collaborative filtering: given a user, kNN finds the closest user that similarly answered other questions and predicts the correctness based on the closest student’s correctness. The core underlying assumption is that if student A has the same correct and incorrect answers on other diagnostic questions as student B, A’s correctness on specific diagnostic questions matches that of student B.

- (a) Complete a function main located at knn.py that runs kNN for different values of k ∈ {1,6,11,16,21,26}. Plot and report the accuracy on the validation data as a function of k.


- (b) Choose k* that has the highest performance on validation data. Report the chosen k∗ and the final test accuracy.


- (c) Implement a function knn_impute_by_item on the same file that performs item-based collaborative filtering instead of user-based collaborative filtering. Given a question, kNN finds the closest question that was answered similarly, and predicts the correctness basted on the closest question’s correctness. State the underlying assumption on item- based collaborative filtering. Repeat part (a) and (b) with item-based collaborative filtering.


- (d) Compare the test performance between user- and item- based collaborative filtering. State which method performs better.

- (e) List at least two potential limitations of kNN for the task you are given.

### 4.2 Item Response Theory.

In this problem, you will implement an Item-Response Theory (IRT) model to predict students’ correctness to diagnostic questions.
The IRT assigns each student an ability value and each question a difficulty value to formulate a probability distribution.

We provide the starter code in part_a/item_response.py.

- (a) Derive the log-likelihood for all students and questions.


- (b) Implement missing functions in item_response.py that performs alternating gradient descent to maximize the log-likelihood. Report the hyperparameters you selected. With your chosen hyperparameters, report the training curve that shows the training and validation log-likelihoods as a function of iteration.


- (c) With the implemented code, report the final validation and test accuracies.


- (d) Select three questions. Using the trained parameters, plot the probability of correctness as a function of student ability for each of the three questions. Comment on the shape of the curves and briefly describe what these curves represent.

### 4.3 Neural Network.

In this problem, you will implement neural networks to predict students’ correctness on a diagnostic question. Specifically, you will design an autoencoder model.

- (a) Describe at least three differences between ALS and neural networks.


- (b) Implement a class AutoEncoder that performs a forward pass of the autoencoder following the instructions in the docstring.


- (c) Train the autoencoder using latent dimensions of k ∈ {10, 50, 100, 200, 500}. Also, tune optimization hyperparameters such as learning rate and number of iterations. Select k* that has the highest validation accuracy.


- (d) With your chosen k*, plot and report how the training and validation objectives changes as a function of epoch. Also, report the final test accuracy.


- (e) Modify a function train so that the objective adds the L2 regularization. 
You may use a method get_weight_norm to obtain the regularization term. Using the k and other hyperparameters selected from part (d), tune the regularization penalty ∈ {0.001, 0.01, 0.1, 1}. With your chosen λ, report the final validation and test accuracy. Does your model perform better with the regularization penalty?

### 4.4 Ensemble.

In this problem, you will be implementing bagging ensemble to improve the stability and accuracy of your base models. Select and train 3 base models with bootstrapping the training set. You may use the same or different base models. Your implementation should be completed in part_a/ensemble.py. To predict the correctness, generate 3 predictions by using the base model and average the predicted correctness. Report the final validation and test accuracy. Explain the ensemble process you implemented. Do you obtain better performance using the ensemble? Why or why not?

## Part B: Novel Methods


In the second part of the project, you will modify one of the algorithms you implemented in part A to hopefully predict students’ answers to the diagnostic question with higher accuracy. In particular, consider the results obtained in part A, reason about what factors are limiting the performance of one of the methods (e.g. overfitting? underfitting? optimization difficulties?) and come up with a proposed modification to the algorithm which could help address this problem. Rigorously test the performance of your modified algorithm, and write up a report summarizing your results as described below.

You will not be graded on how well the algorithm performs (i.e. its accuracy); rather, your grade will be based on the quality of your analysis. Try to be creative! You may also optionally use the provided metadata (question_meta.csv and student_meta.csv) to improve the accuracy of the model. At last, you are free to use any third-party ideas or code as long as it is publicly available. You must properly provide references to any work that is not your own in the write-up.

Your report should be at most 4 pages long (excluding references) and should include the following sections:

1. **Formal Description**: 

 - Precisely define the way in which you are extending the algorithm. You should provide equations and possibly an algorithm box. Describe way your proposed method should be expected to perform better. For instance, are you intending it to improve the optimization, reduce overfitting, etc.?


2. **Figure or Diagram**: 

 - Showing the overall model or idea. The idea is to make your report more accessible, especially to readers who are starting by skimming your report.


3. **Comparison or Demonstration**:

 - A comparison of the accuracies obtained by your model to those from baseline models. Include a table or a plot for an illustrative comparison.


 - Based on the argument you gave for why your extension should help, design and carry out an experiment to test your hypothesis. E.g., consider how you would disentangle whether the benefits are due to optimization or regularization.
 
4. **Limitations**:

 - Describe some settings in which we’d expect your approach to perform poorly, or where all existing models fail.


 - Try to guess or explain why these limitations are the way they are.


 - Give some examples of possible extensions, ways to address these limitations, or open problems.