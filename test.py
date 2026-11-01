import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

X_train = np.array([
    [22, 28000, 1, 55],
    [25, 35000, 3, 62],
    [30, 48000, 5, 70],
    [35, 60000, 8, 78],
    [40, 75000, 12, 85],
    [45, 90000, 16, 90],
    [23, 31000, 2, 58],
    [28, 42000, 4, 65],
    [33, 55000, 7, 74],
    [38, 68000, 10, 82],
    [43, 83000, 14, 88],
    [50, 110000, 22, 95]
], dtype=float)

y_train = np.array([
    0, 0, 0, 1, 1, 1,
    0, 0, 1, 1, 1, 1
])

X_test = np.array([
    [27, 40000, 4, 64],
    [37, 65000, 9, 80],
    [47, 95000, 18, 92],
    [24, 33000, 2, 60]
], dtype=float)

scaler = StandardScaler()
scaling = scaler.fit_transform(X_train)
print(scaler.mean_)
print(scaler.scale_)
print(scaler.var_)
print(scaling[0:3])
test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier()
model.fit(scaling, y_train)
print(model.predict(test_scaled))

scaler_2 = MinMaxScaler()
scaling = scaler_2.fit_transform(X_train)
print(scaling[0:3])
print(scaler_2.data_max_)
print(scaler_2.scale_)
print(scaler_2.data_min_)
print(scaler_2.data_range_)
test_scaled_2 = scaler_2.transform(X_test)

model_2 = KNeighborsClassifier()
model_2.fit(scaling, y_train)
print(model_2.predict(test_scaled_2))