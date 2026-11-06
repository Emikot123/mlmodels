import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing.errors import *

class EmikotKNeighborsClassifier:
    def __init__(self):
        self.x = None
        self.y = None
        self.k = None

    def fit(self, xy:tuple, k: int):
        """
        Here is initialization of model, the data
        and the k chosen
        """
        x, y = (np.asarray(a) for a in xy)
        if x.shape[0] != y.shape[0]:
            raise ArrayLengthError(x, y)
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(x):
            raise ValueError("ERROR: k must be an integer from 1 to the number of points.")
        self.x, self.y, self.k = x, y, k

    def predict(self, x_arr: np.ndarray):
        """
        numpy array of x, y coordinates for multiple predictions. This function is optimized
        for better predictions, by finding if the amount of upvotes is same by classes
        then it finds the class with less distance together from the prediction.
        """
        if not isinstance(x_arr, np.ndarray):
            raise TypeError("ERROR: x_arr must be a numpy array")
        if x_arr.ndim != 2:
            raise InvalidDimensionError("ERROR: x_arr must be 2D")
        if x_arr.shape[1] != self.x.shape[1]:
            raise ValueError("ERROR: x_arr must have the same amount of features as training data")

        results = []
        for arr in x_arr:
            results.append(self.__classify(arr))
        return np.array([results])




    def __classify(self, x: np.ndarray):
        """
        This is main function, a helper one which finds the predictions itself,
        By using Pythagorean Theorem to find distance of each point from prediction.
        After finding we have array of distances, we sort it from smallest to biggest
        and the first (k) elements get chosen, and then out of these points which class appeared
        most it gets to be predicted but if the appearance of class is same for some then we
        choose the sum of those points with less distance
        """
        distances = (np.sqrt(((np.abs(self.x - x))**2).sum(axis=1)))
        indexes = np.argsort(distances)[:self.k]
        near_labels = self.y[indexes]
        near_dists = distances[indexes]

        classes, counts = np.unique(self.y[indexes], return_counts=True)
        total_dist = np.array([near_dists[near_labels == c].sum() for c in classes])
        tied = counts == counts.max()
        winner = np.argmin(np.where(tied, total_dist, np.inf))
        return classes[winner]



    @staticmethod
    def confusion_matrix(y_test, y_pred):
        """
        This one is weird to explain but using train_test_split,
        you have test data and the predicted data and confusion matrix
        shows which ones were predicted and which ones were right
        """

        if not isinstance(y_test, np.ndarray) or not isinstance(y_pred, np.ndarray):
            raise TypeError("ERROR: y_test or y_pred must be a numpy array")
        elif len(y_test) != len(y_pred):
            raise ValueError("ERROR: y_test and y_pred must have the same length")
        elif y_test.ndim != 1 or 1 != y_pred.ndim:
            raise InvalidDimensionError("ERROR: y_test or y_pred must be 1D")

        classes = np.unique(np.concatenate([y_test, y_pred]))

        matrix = np.zeros((len(classes), len(classes)), dtype=int)
        for real, pred in zip(y_test, y_pred):
            row = int(np.where(classes == real)[0][0])
            col = int(np.where(classes == pred)[0][0])
            matrix[row, col] += 1

        return classes, matrix

    def accuracy(self, y_test, y_pred):
        """
        Accuracy of the prediction
        """
        _, matrix = self.confusion_matrix(y_test, y_pred)
        return np.trace(matrix) / matrix.sum()

    def classification_report(self, y_test, y_pred):
        """
        Returns report on your classification predictions and comparisment
        using confusion matrix, it gives tuple with 3 elements: report itself of precision recall f1-score a
        and support for each class then it returns macro_avg, weight_avg of each and last element returns accuracy,
        just know that first 2 elements are dataframe from pandas
        """

        classes, matrix = self.confusion_matrix(y_test, y_pred)
        scores = np.zeros((len(classes), 4))

        scores[:, 0] = self.__prevent_div_by_zero(np.diag(matrix) , np.sum(matrix, axis=0)) #Classes precision
        scores[:, 1] = self.__prevent_div_by_zero(np.diag(matrix) , np.sum(matrix, axis=1))
        scores[:, 2] = self.__prevent_div_by_zero((2*scores[:, 0]*scores[:, 1]) , (scores[:, 0]+scores[:, 1]))
        scores[:, 3] = np.sum(matrix, axis=1)

        accuracy = np.trace(matrix) / np.sum(matrix)
        macro_avg =  np.mean(scores[:, :3], axis=0)
        weighted_avg = self.__prevent_div_by_zero((scores[:, :3].T @ scores[:, 3]) , np.sum(matrix))

        report = pd.DataFrame(
            scores,
            index=classes,
            columns=["precision", "recall", "f1-score", "support"]
        )
        other_stats = pd.DataFrame([macro_avg, weighted_avg],
                                index=["macro_avg", "weighted_avg"],
                                columns=["precision", "recall", "f1-score"])

        return report, other_stats, accuracy

    @staticmethod
    def __prevent_div_by_zero(a, b):
        return np.divide(
            a, b,
            out=np.zeros_like(a, dtype=float),
            where=np.asarray(b) != 0
        )







