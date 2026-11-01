import numpy as np
import matplotlib.pyplot as plt
from preprocessing.errors import *


class EmikotLinearRegression:
    def __init__(self):
        self.x = None
        self.y = None
        self.coeffs = None

    def __find_coefficients(self):
        self.coeffs = np.linalg.inv(self.x.T @ self.x)@(self.x.T@self.y)

    def fit(self, xy: tuple) -> None:
        if not isinstance(xy, tuple):
            raise TypeError("ERROR: Use tuples where first element is 2D Numpy array(features) and second element is Numpy array of label")
        if len(xy) != 2:
            raise WrongTupleSize(xy)
        if not isinstance(xy[0], np.ndarray) or not isinstance(xy[1], np.ndarray):
            raise TypeError("ERROR: Use Numpy arrays")
        if xy[0].ndim != 2:
            raise InvalidDimensionError('ERROR: Features should be 2D numpy array')
        if xy[1].ndim != 1:
            raise InvalidDimensionError('ERROR: Label should be 1D numpy array')
        if xy[0].shape[0] != xy[1].shape[0]:
            raise ValueError("ERROR: Either one of features or label don't have same length as others")

        self.x = self.x = np.column_stack((np.ones(xy[0].shape[0]), xy[0]))
        self.y = xy[1]
        self.__find_coefficients()


    def predict(self, x: np.ndarray):
        if not isinstance(x, np.ndarray):
            raise TypeError("ERROR: Use Numpy arrays")
        if x.ndim != 2:
            raise InvalidDimensionError('ERROR: Features should be 2D numpy array')
        return self.coeffs[0] + np.dot(x ,self.coeffs[1:])


    #TESTING OF MODEL

    def mean_absolute_error(self, xy: tuple) -> float:
        if len(xy) != 2:
            raise WrongTupleSize(xy)
        predictions = self.predict(xy[0])
        return np.mean(np.abs(predictions - xy[1]))

    def mean_squared_error(self, xy: tuple):
        if len(xy) != 2:
            raise WrongTupleSize(xy)
        predictions = self.predict(xy[0])
        results = (predictions - xy[1]) **2
        return 1 / len(xy[0]) * sum(results)


    def root_mean_squared_error(self, xy: tuple):
        if len(xy) != 2:
            raise WrongTupleSize(xy)
        return np.sqrt(self.mean_squared_error(xy))

    def r_squared(self, xy: tuple) -> float:
        if len(xy) != 2:
            raise WrongTupleSize(xy)

        lazy_guess = np.mean(xy[1])
        sum_distance = xy[1] - lazy_guess
        bottom_total = 0
        for num in sum_distance:
            bottom_total += (num ** 2)
        top_total = self.mean_squared_error(xy) * len(xy[1])
        result = 1 - top_total / bottom_total
        return result
