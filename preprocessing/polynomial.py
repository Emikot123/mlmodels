import numpy as np
from itertools import combinations_with_replacement
from errors import *

class EmikotPolynomialFeatures:
    def __init__(self, degrees: int):
        if not isinstance(degrees, int):
            raise TypeError("ERROR: degree must be an integer")
        if degrees < 0:
            raise ValueError("ERROR: degree should be > 0")

        self.data = None
        self.degrees = degrees
        self.combs = None
        self.n_features = None

    def fit(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError("ERROR: data must be a numpy array")
        if data.ndim != 2:
            raise InvalidDimensionError("ERROR: data must be a 2D numpy array")
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("ERROR: data must not be empty")


        self.data = data
        self.n_features = data.shape[1]
        all_combs = []
        features = range(data.shape[1])
        for degree in range(1, self.degrees + 1):
            all_combs.extend(list(combinations_with_replacement(features, degree)))
        self.combs = all_combs

    def transform(self, data: np.ndarray):
        if self.data is None:
            raise RuntimeError("ERROR: First you must fit the data")
        if not isinstance(data, np.ndarray):
            raise TypeError("ERROR: data must be a numpy array")
        if data.ndim != 2:
            raise InvalidDimensionError("ERROR: data must be a 2D numpy array")
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("ERROR: data must not be empty")
        columns = []
        for comb in self.combs:
            result = np.prod(data[:, comb], axis=1)
            columns.append(result)
        return np.column_stack(columns)

    def fit_transform(self, data: np.ndarray):
        self.fit(data)
        return self.transform(data)

X = np.array([[2, 3, 4, 5],
              [4, 6, 7, 2],
              [8, 3, 1, 5]])

poly = EmikotPolynomialFeatures(degrees=3)
poly.fit(X)

