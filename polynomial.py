import numpy as np
from errors import *
import matplotlib.pyplot as plt

class EmikotPolynomialRegression:
    def __init__(self):
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.slope = None
        self.intercept = None
        self.bend = None

    def fit(self, xy: tuple):
        if not isinstance(xy, tuple):
            raise TypeError("ERROR: It should be tuple of 2 numpy arrays, x and y")
        elif not isinstance(xy[0], np.ndarray) or not isinstance(xy[1], np.ndarray):
            raise TypeError('ERROR: Should be numpy array')
        elif len(xy) != 2:
            raise WrongTupleSize(xy)
        elif len(xy[0]) != len(xy[1]):
            raise ArrayLengthError(xy[0], xy[1])

        self.x = xy[0]
        self.y = xy[1]


    def polynomial(self):
        prep = [self.x**2, self.x**3, self.x**4, self.x * self.y, (self.x**2) * self.y]
        n = len(self.x)
        sums = np.sum(prep, axis=1)
        equations = np.array([
            [n, self.x.sum(), sums[0], self.y.sum()],
            [self.x.sum(), sums[0], sums[1], sums[3]],
            [sums[0], sums[1], sums[2], sums[4]]
        ])
        coefficients
        r2 = equations[1] - equations[0]
        r3 = equations[2] - equations[0]
        print(r2, r3)

x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y1 = np.array([1, 4, 9, 16, 25, 36, 49, 64])
model = EmikotPolynomialRegression()
model.fit((x1,y1))
model.polynomial()

