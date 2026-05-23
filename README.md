# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create the dataset for Logistic Regression.
2. Apply feature scaling and initialize weight, bias, learning rate, and epochs.
3. Use the sigmoid function and Gradient Descent to train the model by updating parameters.
4. Make predictions, display the results, and plot the loss graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: JAYASHREE J
RegisterNumber:  212225040145

# Implementation of Logistic Regression using Gradient Descent

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Sample Dataset
# -----------------------------
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# -----------------------------
# Step 2: Feature Scaling
# -----------------------------
X = (X - np.mean(X)) / np.std(X)

# -----------------------------
# Step 3: Initialize Parameters
# -----------------------------
w = 0
b = 0
learning_rate = 0.1
epochs = 1000
n = len(X)

# -----------------------------
# Sigmoid Function
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------------
# Step 4: Gradient Descent
# -----------------------------
losses = []

for i in range(epochs):

    # Linear equation
    z = w * X.flatten() + b

    # Apply sigmoid
    y_hat = sigmoid(z)

    # Binary Cross Entropy Loss
    loss = -(1/n) * np.sum(y * np.log(y_hat + 1e-9) +
                           (1-y) * np.log(1-y_hat + 1e-9))

    losses.append(loss)

    # Gradients
    dw = (1/n) * np.sum((y_hat - y) * X.flatten())
    db = (1/n) * np.sum(y_hat - y)

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

# -----------------------------
# Step 5: Predictions
# -----------------------------
predictions = sigmoid(w * X.flatten() + b)
predicted_classes = [1 if i >= 0.5 else 0 for i in predictions]

# -----------------------------
# Step 6: Results
# -----------------------------
print("Final Weight:", w)
print("Final Bias:", b)
print("Predicted Classes:", predicted_classes)

# -----------------------------
# Step 7: Plot Loss Graph
# -----------------------------
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss vs Iterations")
plt.show()
*/
```

## Output:
<img width="1008" height="799" alt="image" src="https://github.com/user-attachments/assets/20b4201c-8013-4d62-a410-9094ddab3453" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

