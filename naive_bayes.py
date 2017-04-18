"""
Bradley Robinson

"""
import numpy as np
from classifier import Classifier


class NaiveBayes(Classifier):
    """
    Naive Bayes classifier for the classification of binary indices of text data (for now). That means that features
    that do not have integers will not work as expected.
    """
    def __init__(self):
        Classifier.__init__(self)
        self.class_priors = None
        self.conditional_probabilities = None
        self.class_indices = {}
        self.class_term_probability = {}

    def fit(self, X, y):
        """

        Parameters:
            X: np.array, must be integers or strings. This is fundamental to this implementation of Naive Bayes
            y: np.array, array of class labels

        Returns:
            None
        """
        self.X = X
        self.y = y
        self.unique_labels = np.unique(self.y)
        self._find_class_priors()
        self._index_classes()

    def _find_class_priors(self):
        """
        Initialized self.class_priors with the frequency that each class occurs.

        Returns:
             None
        """
        counts = np.unique(self.y, return_counts=True)
        self.class_priors = {counts[0][i]: counts[1][i]/self.y.shape[0] for i in range(len(counts[0]))}

    def _index_classes(self):
        """
        Simply finds the indices for each class so they can be used in the future.

        Returns:
            None
        """
        for c in self.unique_labels:
            self.class_indices[c] = self.y[self.y == c]

    def _train_conditional_probabilities(self):
        """

        """

        for c in self.unique_labels:
            c_data = self.X[self.class_indices[c]]
            c_total_freq = np.sum(c_data)
            c_probabilites = {}
            for col in range(c_data.shape[1]):
                col_sum = np.sum(c_data[:,col])
                # TODO: Start here
                # col_probability
        # a. Goes through and finds how likely each column is in each class


    def _find_conditional_probabilites(self, observation):
        """
        Used to find the conditional probability of each term occurring in an observation.

        Parameters:
            observation: np.array, an observation/row of data.

        Returns:

        """

        pass

    def predict(self, X):
        """

        :param X:
        :return:
        """
        for row in X:
            self._find_conditional_probabilites()
        pass