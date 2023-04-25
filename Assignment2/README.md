# CSC311 - Assignment2

## 1. Expected Loss and Bayes Optimality


You are running an email service, and one of your key features is a spam filter. Every email is either spam or non-spam, which we represent with the target t ∈ {Spam, NonSpam}. You need to decide whether to keep it in the inbox or remove it to the spam folder. We represent this with the decision variable y ∈ {Keep, Remove}. We’d like to remove spam emails and keep non-spam ones, but the customers will be much more unhappy if we remove a non-spam email than if we keep a spam email.

Your studies indicate that 20% of the emails are spam, i.e. Pr(t = Spam) = 0.2.

- (a) Evaluate the expected loss E[J (y, t)] for the policy that keeps every email (y = Keep), and for the policy that removes every email (y = Remove).

- (b) Now suppose you get to observe a feature vector x for each email, and using your knowledge of the joint distribution p(x,t), you infer p(t|x). Determine how you will make Bayes optimal decision y∗ ∈ {Keep, Remove} given the conditional probability Pr(t = Spam | x).

- (c) After some analysis, you found two words that are indicative of an email being spam: “Sale” and “Prince”. You define two input features: x1 = 1 if the email contains the word “Sale” and x1 = 0 otherwise, and x2 = 1 if the email contains the word “Prince” and x2 = 0 otherwise.

- (d) What is the expected loss E[J(y∗,t)] for the Bayes optimal decision rule from part (c)?

## 2. Feature Maps.
Suppose we have the following 2-D data-set for binary classification.

- (a) Explain why this data-set is NOT linearly separable in a few sentences.

- (b) Suppose you are interested in studying if the above data-set can be separable by a quadratic functions y = w1x1 + w2x2 + w3x2. Write all the constraints that w1, w2, w3 have to satisfy. You do not need to solve for the w using the constraints.

## 3. kNN vs. Logistic Regression.
In this problem, you will compare the performance and characteristics of two classifiers: k-Nearest Neighbors and Logistic Regression. You will complete the provided code in q3/ and run experiments with the completed code. You should understand the code instead of using it as a black box.

- (a) k-Nearest Neighbors. Use the supplied kNN implementation to predict labels for mnist_valid, using the training set mnist_train.

- (b) Logistic Regression. Read the provided code in run_logistic_regression.py and logistic.py.
