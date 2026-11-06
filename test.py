from sklearn.model_selection import train_test_split
from preprocessing.st_scalar import EmikotStandardScaler
import pandas as pd
import numpy as np
from classification import EmikotKNeighborsClassifier

data = pd.read_csv("students_2.csv")

features = data[["hours_slept", "hours_studied"]].to_numpy()
label = data["grade"].to_numpy()


train_x, test_x, train_y, test_y = train_test_split(features, label, test_size=0.2, random_state=42)

model = EmikotKNeighborsClassifier()
scaler = EmikotStandardScaler()

scaled = scaler.fit_transform(train_x)

model.fit((scaled, train_y), k=6)

print(model.predict(np.array([[5, 6], [7, 8]])))

