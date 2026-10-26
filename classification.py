import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from errors import *

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


    def visualise(self, steps = 300):

        fig, ax = plt.subplots()

        classes, z_nums = np.unique(self.z, return_inverse=True)

        vis_features = np.c_[self.x, self.y]

        x_range = np.linspace(self.x.min() - 0.5, self.x.max() + 0.5, steps)
        y_range = np.linspace(self.y.min() - 0.5, self.y.max() + 0.5, steps)

        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = self.predict(grid_points)

        pred_nums = np.searchsorted(classes, predictions)
        borders = pred_nums.reshape(xx.shape)

        levels = np.arange(len(classes) + 1) - 0.5

        ax.contourf(xx, yy, borders, levels=levels, cmap='viridis', alpha=0.3)
        points = ax.scatter(self.x, self.y, c=z_nums, cmap='viridis',
                            vmin=0, vmax=len(classes) - 1, edgecolors='black')
        ax.legend(points.legend_elements()[0], classes, title='class')
        plt.tight_layout()
        plt.show()


    @staticmethod
    def confusion_matrix(y_test, y_pred):
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
        



data = pd.read_csv('students_2.csv')
features = data[["hours_studied", "hours_slept"]].to_numpy()
result = data["grade"].to_numpy()
train_x, test_x, train_y, test_y = train_test_split(features, result, test_size=0.2, random_state=32)

hours_stud = train_x[:, 0]
hours_slept = train_x[:, 1]
emik_train = (hours_stud, hours_slept, train_y)
model = EmikotKNeighborsClassifier()
model.fit(emik_train, 7)
pred = model.predict(test_x)
model.visualise()
