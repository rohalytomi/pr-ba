import numpy as np
import matplotlib.pyplot as plt

def visualize_data(x, y):
    plt.scatter(x, y)
    plt.xlabel("Feature (x)")
    plt.ylabel("Target (y)")
    plt.title("Synthetic Data with Polynomial Relationship and Noise")
    plt.show()

def main():
    coefficients = [1, 0.02, -0.002, 0.014]
    x_values = np.linspace(-10, 10, 100)
    x, y = generate_synthetic_data(x_values, coefficients)
    visualize_data(x, y)