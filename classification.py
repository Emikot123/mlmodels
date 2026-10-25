import numpy as np
import matplotlib.pyplot as plt
from errors import *
from itertools import islice
from collections import Counter

class EmikotKNeighborsClassifier:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.k = None

    def fit(self, xy_z:tuple, k: int):
        x, y, z = (np.asarray(a) for a in xy_z)
        if not (len(x) == len(y) == len(z)):
            raise ArrayLengthError(x, z)
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(x):
            raise ValueError("ERROR: k must be an integer from 1 to the number of points.")
        self.x, self.y, self.z, self.k = x, y, z, k



    def predict(self, xy_arr: np.ndarray | tuple):
        if self.x is None:
            raise RuntimeError("ERROR: You must call fit before calling predict.")

        if isinstance(xy_arr, tuple):
            if len(xy_arr) != 2:
                raise WrongTupleSize()
            prediction = self.__classify(xy_arr)
            return prediction
        if isinstance(xy_arr, np.ndarray):
            if xy_arr.ndim != 2:
                raise InvalidDimensionError("ERROR: Predictions array must have 2 dimensions.")
            if xy_arr.shape[1] != 2:
                raise ArrayColumnError("ERROR: Predictions array must have 2 columns.")
            results = []
            for arr in xy_arr:
                results.append(self.__classify(arr))
            return np.array(results)
        raise TypeError("ERROR: Your prediction is not tuple(single prediction) neither numpy array(For more than 1 predictions)")


    def __classify(self, xy: np.ndarray | tuple):

        distances = np.sqrt((self.x - xy[0]) ** 2 + (self.y - xy[1]) ** 2)
        nearest = np.argsort(distances)[:self.k]
        near_labels = self.z[nearest]
        near_dists = distances[nearest]

        classes, votes = np.unique(near_labels, return_counts=True)
        total_dist = np.array([near_dists[near_labels == c].sum() for c in classes])
        tied = votes == votes.max()
        winner = np.argmin(np.where(tied, total_dist, np.inf))
        return classes[winner]

