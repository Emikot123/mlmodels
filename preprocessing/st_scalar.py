import numpy as np
from errors import *


class EmikotStandardScaler:
    def __init__(self):
        self.data = None
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError('ERROR: Use Numpy arrays')
        if data.ndim != 2:
            raise InvalidDimensionError('ERROR: Use 2D Numpy arrays')

        self.data = data
        self.mean = np.mean(self.data, axis=0)
        stds = np.std(self.data, axis=0)
        self.std = np.where(stds == 0, 1, stds)

    def transform(self, data: np.ndarray):
        if self.data is None:
            raise RuntimeError('ERROR: First fit the data')
        if not isinstance(data, np.ndarray):
            raise TypeError('ERROR: Use Numpy arrays')
        if data.ndim != 2:
            raise InvalidDimensionError('ERROR: Use 2D Numpy arrays')
        if data.shape[1] != self.data.shape[1]:
            raise ValueError('ERROR: Number of features does not match')

        result = (data - self.mean) / self.std
        return result

    def fit_transform(self, data: np.ndarray):
        self.fit(data)
        return self.transform(data)

