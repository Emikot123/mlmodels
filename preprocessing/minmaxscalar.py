import numpy as np
from errors import *

class EmikotMinMaxScaler:
    def __init__(self, range: tuple = (0, 1)):
        if not isinstance(range, tuple):
            raise TypeError('ERROR: range is not a tuple')
        if len(range) != 2:
            raise ValueError('ERROR: range should have 2 elements: Min and Max')
        if range[0] >= range[1]:
            raise ValueError('ERROR: first elements should be smaller than second')

        self.data = None
        self.data_min = None
        self.data_max = None
        self.range = range

    def fit(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError('ERROR: Use Numpy arrays')
        if data.ndim != 2:
            raise InvalidDimensionError('ERROR: Use 2D Numpy arrays')

        self.data = data
        self.data_min = np.min(data, axis=0)
        self.data_max = np.max(data, axis=0)

    def transform(self, data: np.ndarray):
        if self.data is None:
            raise RuntimeError('ERROR: First fit the data')
        if not isinstance(data, np.ndarray):
            raise TypeError('ERROR: Use Numpy arrays')
        if data.ndim != 2:
            raise InvalidDimensionError('ERROR: Use 2D Numpy arrays')
        if data.shape[1] != self.data.shape[1]:
            raise ValueError('ERROR: Number of features does not match')

        max_min = self.data_max - self.data_min
        max_min_diff = np.where(max_min == 0, 1, max_min)
        return self.range[0] + ((data - self.data_min) / max_min_diff) * (self.range[1] - self.range[0])

    def fit_transform(self, data: np.ndarray):
        self.fit(data)
        return self.transform(data)
