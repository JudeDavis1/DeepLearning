import re
import random
import numpy as np
import pandas as pd

# utilities for making training easier and more clean
from utils import sigmoid
from sklearn.model_selection import train_test_split


def sigmoid(x, derivative=False):
    standard_sigmoid = 1/(1 + np.exp(-x)) # actual sigmoid function
    if derivative==True:
        return standard_sigmoid * (1 - standard_sigmoid) # sigmoid derivative
    return standard_sigmoid

class NeuralNetwork:

    def __init__(self, lr, epochs, x, y):
        self.lr = lr
        self.epochs = epochs
        self.x_train = x
        self.y_train = y
        self.weights = np.random.random(x_train.shape)
        self.bias = 0

        random.seed(1)

    def fit():
        n_samples, n_features = x.shape

        for _ in range(self.epochs):
            prediction = sigmoid(np.dot(self.x, self.weights) + self.bias)

            # back-propagation
            y_weights = (1 / n_samples) * np.dot(self.x.T, (prediction - self.y))
            y_bias = (1 / n_samples) * np.sum(prediction - self.y)

            # update weights and biases
            self.weights -= self.lr * y_weights[0]
            self.weights -= self.lr * y_weights[1]
            self.bias -= self.lr * y_bias

    def predict(x, y):
    	ret = np.dot(x, self.weights) + self.bias

    	return ret



df = pd.read_csv("property_sales_transactions.csv", low_memory=False, dtype=str)

X = df["PRICE"].head(100)
Y = df["PRICE"].tail(100)

x_train, x_test, y_train, y_test = train_test_split(X, Y)


model = NeuralNetwork(lr=0.01, epochs=10000, x=x_train, y=y_train)

print("Neural Network is learning")
model.fit()

print(model.predict(x_test, y_test))