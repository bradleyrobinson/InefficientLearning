"""
Created by Bradley Robinson

This is meant as a way of exploring machine learning, not as an effective tool for machine learning. If you're
interested in more serious projects, please take a look at tools such as sklearn and tensorflow. This is probably not
the most efficient implementation.

Base functionality for classifiers.
"""
import numpy as np


class Classifier(object):
    """

    """
    def __init__(self):
        """

        """
        self.X = None
        self.y = None
        self.unique_labels = None

    def fit(self, X, y):
        """
        Function that must be implemented in classes inheriting from classifier. Takes training data and fits the model
        for predictions.

        Parameters:
            X: np array, feature data to be used for model training
            y: np array, classification labels for training data

        Returns:
            None
        """
        pass

    def predict(self, X):
        """
        Makes predictions using features in data. Must have the same features as the data used for fitting.

        Parameters:
            X:

        Returns:
             np.array: prediction labels.
        """
        pass

    def score(self, predictions, true_labels):
        pass