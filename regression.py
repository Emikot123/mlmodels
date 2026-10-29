import numpy as np
import matplotlib.pyplot as plt

from classification import EmikotKNeighborsClassifier
from errors import *


class EmikotLinearRegression:
    def __init__(self):
        self.x = None
        self.y = None
        self.slope = None
        self.intercept = None
        self.average_point = None


    def __find_average_point(self):
        """
        Function that finds average point of points for Linear Regression,
        it needs two array with same length for x and y points without numpy
        """
        if len(self.x) != len(self.y):
            raise ArrayLengthError(self.x, self.y)
        n = len(self.x)
        x_point = (1 / n) * sum(self.x)
        y_point = (1 / n) * sum(self.y)
        coordinate = (x_point, y_point)
        self.average_point = coordinate
        return coordinate

    def __find_slope(self) -> float:
        """
        Finds slope by complex formula rewritten from scratch
        First it handles error like it should be numpy array and have same length
        then it does the formulas with sigma notations to find dividend and divisor and divides to find slope
        """
        if not (isinstance(self.x, np.ndarray) and isinstance(self.y, np.ndarray)):
            raise TypeError("ERROR: Use numpy arrays")
        if len(self.x) != len(self.y):
            raise ArrayLengthError(self.x, self.y)
        top_total = 0
        for x_num, y_num in zip(self.x, self.y):
            top_action = (x_num - self.average_point[0]) * (y_num - self.average_point[1])
            top_total += top_action
        bottom_total = 0
        for x_num in self.x:
            bottom_action = (x_num - self.average_point[0])
            bottom_total += (bottom_action ** 2)
        if bottom_total == 0:
            raise ValueError("ERROR: All x values are the same, so the slope is undefined")
        self.slope = top_total / bottom_total
        return top_total / bottom_total

    def __find_intercept(self) -> float:
        """
        Using slope and average_point we reverse the formula
        From:
        y = mx + b
        To:
        b = y - mx
        Its that simple
        """
        y_intercept = self.average_point[1] - self.slope * self.average_point[0]
        self.intercept = y_intercept
        return y_intercept

    def fit(self, xy: tuple) -> None:
        self.x = xy[0]
        self.y = xy[1]
        self.__find_average_point()
        self.__find_slope()
        self.__find_intercept()


    def predict(self, x):
        result = self.slope * x + self.intercept
        return result

    def visualise(self):
        fig, ax = plt.subplots()
        ax.scatter(self.x, self.y, color='blue')
        line = self.predict(self.x)
        ax.plot(self.x, line, color='red')
        plt.tight_layout()
        plt.show()

    #TESTING OF MODEL

    def mean_absolute_error(self, xy: tuple) -> float:
        if len(xy) != 2:
            raise WrongTupleSize(xy)
        predictions = self.predict(xy[0])
        results = np.abs(predictions - xy[1])
        return 1 / len(xy[0]) * sum(results)

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

x_1 = np.array([0.0, 1.0, 1.5, 5.0, 6.0, 7.0])
x_2 = np.array([0.0, 1.0, 2.0, 5.0, 5.5, 6.0])
tup = (x_1, x_2)
model = EmikotLinearRegression()
model.fit(tup)
x = [1,4,6,7]