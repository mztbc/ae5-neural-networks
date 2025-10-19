import pandas
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pandas.read_csv("dataset.csv")

X = dataset.drop(columns=["status"])
y = dataset["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
