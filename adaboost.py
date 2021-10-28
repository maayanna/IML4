"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np
import ex4_tools
import matplotlib.pyplot as plt

VALT = [5,10,50,100,200,500]


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        D = np.ones(len(X)) / len(y)

        for t in range(self.T):

            self.h[t] = self.WL(D, X, y)
            error = np.dot(D, self.h[t].predict(X) != y)
            self.w[t] = 0.5 * np.log(1/error - 1)
            numerator = D * (np.exp(-self.w[t] * y * self.h[t].predict(X)))
            D = numerator / np.sum(numerator)

        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        y_predict = np.zeros(len(X))

        for i in range(max_t):

            y_predict += self.w[i] * self.h[i].predict(X)

        y_predict[y_predict > 0] = 1
        y_predict[y_predict <= 0] = -1

        return y_predict

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the correct predictions when predict only with max_t weak learners (float)
        """
        return np.sum((self.predict(X, max_t)) != y) / y.size


def task10(my_adaboost, X, y, test_x, test_y, noise):
    """
    Use it to generate 5000
    samples without noise (i.e. noise ratio=0). Train an Adaboost classifier over this data.
    Use the DecisionStump weak learner mentioned above, and T “ 500. Generate another
    200 samples without noise (”test set”) and plot the training error and test error, as a
    function of T. Plot the two curves on the same figure.
    :return:
    """

    error_test = list()
    error_train = list()

    for i in range(500):

        error_test.append(my_adaboost.error(test_x, test_y, i + 1))
        error_train.append(my_adaboost.error(X, y, i + 1))

    # plt.plot(error_train, label = "TRAIN ERRROR ")
    # plt.plot(error_test, label = "TEST ERROR")
    # plt.xlabel("T")
    # plt.ylabel("ERROR")
    # plt.title("Q10 : ERROR OF TEST AND TRAIN DEPENDING ON T, NOISE = " + str(noise))
    # plt.legend()
    # plt.show()

# task10()

def task11(X, y, test_x, test_y, noise):
    """
    Plot the decisions of the learned classifiers with T P t5, 10, 50, 100, 200, 500u together with
    the test data. You can use the function decision boundaries together with plt.subplot for
    this purpose.
    :return:
    """
    my_error = list()
    d = list()

    for index, val in enumerate(VALT):

        my_adaboost = AdaBoost(ex4_tools.DecisionStump, val)
        # plt.subplot(2, 3, index + 1)
        d.append(my_adaboost.train(X, y))
        ex4_tools.decision_boundaries(my_adaboost, test_x, test_y, val)
        my_error.append(my_adaboost.error(test_x, test_y, val))

    # plt.suptitle("Q11 - Decisions of the learned classifiers with T. NOISE = " + str(noise))
    # plt.savefig("Q11-1.pdf")
    # plt.show()

    return my_error, d

# task11()

def task12(my_adaboost, X, y, my_error, noise):
    """
    Out of the different values you used for T, find Tˆ, the one that minimizes the test error.
    What is Tˆ and what is its test error? Plot the decision boundaries of this classifier together
    with the training data.
    :return:
    """

    min_index = np.argmin(my_error)
    t = VALT[min_index]
    ex4_tools.decision_boundaries(my_adaboost, X, y, t)
    # plt.title("Q12 - Classifier with T = " + str(t) + ", NOISE = " + str(noise) + " and ERROR = " + str(my_error[min_index]))
    # plt.show()

# task12()

def task13(my_adaboost, X, y, d, noise):
    """
    Look into the AdaBoost: Take the weights of the samples in the last iteration of the
    training (DT). Plot the training set with size proportional to its weight in DT
    , and color
    that indicates its label (again, you can use decision boundaries). Oh! we cannot see any
    point! the weights are to small... so we will normalize them: D = D / np.max(D) * 10.
    What do we see now? can you explain it?
    :return:
    """

    for index, t in enumerate(VALT):

        ex4_tools.decision_boundaries(my_adaboost, X, y, t, (d[index]/max(d[index])) * 10 )
        plt.suptitle("Training set with size proportional to its weight, NOISE = " + str(noise))

    plt.show()

# task13()

def ex4():

    for noise in [0, 0.01, 0.4]:

        X, y = ex4_tools.generate_data(5000, noise)
        test_x, test_y = ex4_tools.generate_data(200, noise)

        my_adaboost = AdaBoost(ex4_tools.DecisionStump, 500)
        my_adaboost.train(X, y)

        #Task 10 for all noise
        task10(my_adaboost,X, y, test_x, test_y, noise)

        #Task 11 for all  noise
        error, d = task11(X, y, test_x, test_y, noise)

        #Task 12 for all noise
        task12(my_adaboost,X, y, error, noise)

        #Task 13 for all noise
        task13(my_adaboost, X, y, d, noise)


# ex4()