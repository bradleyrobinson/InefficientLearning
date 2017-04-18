"""
By Bradley Robinson

Like perceptron.py, this is not truly original, but an adaptation of Sebastian Raschka's implementation in the book
Python Machine Learning
"""
import numpy as np


class Adaline(object):
    """
    ADaptive LInear NEuron classifier
    """
    def __init__(self, eta, num_iters = 50):
        self.eta = eta
        self.num_iters = num_iters

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.num_iters):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0:] += self.eta * errors.sum()
            cost = (errors**2).sum(errors)
            self.cost_.append(cost)
        return self


    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
