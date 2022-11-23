https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
   This file contains the Linear Regression Regressor
   and the Logistic Regression classifier

   Brown CS142, Spring 2019
'''
import random
import numpy as np


def l2_loss(predictions, Y):
    '''
    Computes L2 loss (sum squared loss) between true values, Y, and predictions.

    @params:
        Y: A 1D Numpy array with real values (float64)
        predictions: A 1D Numpy array of the same size of Y
    @return:
        L2 loss using predictions for Y.
    '''
    # TODO
    pass


def softmax(x):
    '''
    Apply softmax to an array

    @params:
        x: the original array
    @return:
        an array with softmax applied elementwise.
    '''
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


class LinearRegression:
    '''
    LinearRegression model that minimizes squared error using either
    stochastic gradient descent or matrix inversion.
    '''
    def __init__(self, n_features, sgd=False):
        '''
        @attrs:
            n_features: the number of features in the regression problem
            sgd: Boolean representing whether to use stochastic gradient descent
            alpha: The learning rate used in SGD
            weights: The weights of the linear regression model.
        '''
        self.n_features = n_features + 1  # An extra feature added for the bias value
        self.sgd = sgd
        self.alpha = 0.2  # Tune this parameter
        self.weights = np.zeros(n_features + 1)

    def train(self, X, Y):
        '''
        Trains the LinearRegression model weights using either
        stochastic gradient descent or matrix inversion.

        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            None
        '''
        if self.sgd:
            self.train_sgd(X, Y)
        else:
            self.train_solver(X, Y)

    def train_sgd(self, X, Y):
        '''
        Trains the LinearRegression model weights until convergence
        using stochastic gradient descent.

        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            None. You can change this to return whatever you want, e.g. an array of loss
            values, to produce data for your project report.
        '''
        # TODO
        pass

    def train_solver(self, X, Y):
        '''
        Trains the LinearRegression model by finding the optimal set of weights
        using matrix inversion.

        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            None
        '''
        # TODO
        pass

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        # TODO
        pass

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            A float number which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            A float number which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]


class LogisticRegression:
    '''
    Multinomial Linear Regression that learns weights by minimizing
    mean squared error using stochastic gradient descent.
    '''
    def __init__(self, n_features, n_classes):
        '''
        Initializes a LogisticRegression classifer.

        @attrs:
            n_features: the number of features in the classification problem
            n_classes: the number of classes in the classification problem
            weights: The weights of the Logistic Regression model
            alpha: The learning rate used in stochastic gradient descent
        '''
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros((n_features + 1, n_classes))  # An extra row added for the bias
        self.alpha = 0.2  # tune this parameter

    def train(self, X, Y):
        '''
        Trains the model, using stochastic gradient descent

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            None. You can change this to return whatever you want, e.g. an array of loss
            values, to produce data for your project report.
        '''
        # TODO
        pass

    def predict(self, X):
        '''
        Compute predictions based on the learned parameters and examples X

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        # TODO
        pass

    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        # TODO
        pass
