import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from main import generate_synthetic_data

def fit_linear_regression(x,y):
    lr=LinearRegression()
    lr.fit(x.reshape(-1,-1),y)
    return lr

def main():
    coefficients = [100, 0.02, -0.002, 0.014]
    x_values = np.linspace(-10, 10, 100)
    x, y = generate_synthetic_data(x_values, coefficients)
    lr=fit_linear_regression(x,y)
    visualize_data_and_fit(x, y, lr)