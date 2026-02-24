# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed.

2.Scale the features using a standard scaler to normalize the data.

3.Initialize model parameters (theta) and add an intercept term to the feature set.

4.Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function.

5.Make predictions on new data by transforming it using the same scaling and encoding applied to the training data. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: UDHAYDHARSHAN S
RegisterNumber:  212225230286
*/
```
```

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("C:/Users/acer/Downloads/Placement_Data (1).csv")


data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})


X = data[["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]].values
y = data["status"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std


n_features = X_train.shape[1]
w = np.zeros(n_features)
b = 0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


lr = 0.01
epochs = 1000

for i in range(epochs):
    z = np.dot(X_train, w) + b
    y_pred = sigmoid(z)

    dw = (1/len(X_train)) * np.dot(X_train.T, (y_pred - y_train))
    db = (1/len(X_train)) * np.sum(y_pred - y_train)

    w -= lr * dw
    b -= lr * db


def predict(X):
    z = np.dot(X, w) + b
    probs = sigmoid(z)
    return (probs >= 0.5).astype(int)


y_test_pred = predict(X_test)

accuracy = np.mean(y_test_pred == y_test)
print("Accuracy:", accuracy)
print("confusion matrix:",classification_report(y_test,y_test_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

```
## Output:
<img width="859" height="204" alt="Screenshot 2026-02-24 094254" src="https://github.com/user-attachments/assets/6d5e3819-30e8-45bb-9167-cb31400a0f5e" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

