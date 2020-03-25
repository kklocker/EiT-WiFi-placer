import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def plot_grid(x, y):
    x, y = np.meshgrid(x, y)
    plt.scatter(x, y)
    plt.show()


def normalize(x, low, high):
    x -= low
    x = x*(high/max(x))
    return x.astype(int)


def gauss_grid(minx, maxx, miny, maxy, n=20):
    """
    Generates a 2D grid of points with a kind of Gaussian distribution, in the intervals given by min,max.
    """
    h = 100
    h_min = 1
    rho = signal.gaussian(n, n/4, sym = False)
    x_array = np.zeros(n)

    for i in range(1, n):
        h_next = max(h_min, h/rho[i])
        x_array[i] = x_array[i-1] + h_next
    y_array = np.copy(x_array)

    x_array = normalize(x_array, minx, maxx)
    y_array = normalize(y_array, miny, maxy)
    return x_array, y_array

