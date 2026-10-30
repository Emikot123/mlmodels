import numpy as np
from errors import *
import matplotlib.pyplot as plt

class EmikotPolynomialRegression:
    def __init__(self):
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.b0 = None
        self.b1 = None
        self.b2 = None

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
        self.polynomial()


    def polynomial(self):
        prep = [self.x**2, self.x**3, self.x**4, self.x * self.y, (self.x**2) * self.y]
        n = len(self.x)
        sums = np.sum(prep, axis=1)
        equations = np.array([
            [n, self.x.sum(), sums[0], self.y.sum()],
            [self.x.sum(), sums[0], sums[1], sums[3]],
            [sums[0], sums[1], sums[2], sums[4]]
        ], dtype=float)
        equations[1] -= (equations[1,0] / equations[0,0])*equations[0]
        equations[2] -= (sums[0] / equations[0,0])*equations[0]
        equations[2] -= (equations[2,1] / equations[1,1])*equations[1]
        self.b2 = equations[2,3] / equations[2,2]
        self.b1 = (equations[1,3] - (equations[1,2]*self.b2)) / equations[1,1]
        self.b0 = (equations[0,3] - equations[0,1]*self.b1 - equations[0,2]*self.b2)/ equations[0,0]

    def predict(self, x):
        arr = []
        for i in x:
            arr.append(self.b0 + self.b1*i + self.b2*(i**2))
        return arr
    def visualise(self):
        fig, ax = plt.subplots()
        line = self.predict(self.x)
        ax.plot(line, color='blue')
        ax.scatter(self.x, self.y, color='red')
        plt.show()

x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y1 = np.array([1, 4, 9, 16, 25, 36, 49, 64])
model = EmikotPolynomialRegression()
model.fit((x1,y1))
model.visualise()
print(pred)

