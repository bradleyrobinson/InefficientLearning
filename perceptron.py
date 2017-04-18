"""
Implementation of a perceptron, much of this code is inspired by Sebastian Raschka's "Python Machine Learning."
"""
import numpy as np


class Perceptron(object):
    """

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit training data

        Parameters:
            X: array-like, shape = [n_samples, n_features]
            y: array-like, shape = n_samples

        Returns:
            Object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        Calculates net input
        Parameters:
            X:
        Returns:
            np.array, vector-like structure with net_input
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Predicts labels
        Parameters:
            X:
        Returns:
            np.array of predicted labels.
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)



