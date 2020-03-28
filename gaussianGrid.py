import numpy as np
from scipy import signal


def normalize(x, low, high):
    x -= low
    x = x*(high/max(x))
    return x.astype(int)


def wall_filter(points, img):
    """
    Filters away points that are inside walls. Works by checking where the refractive index is not 1.
    """
    deletion_mask = img[points[:, 0], points[:, 1]] != 1
    filtered_points = points[~deletion_mask]
    return filtered_points


def gauss_grid(img, n=20):
    """
    Generates a 2D grid of n*n points on the image with some sort of Gaussian distribution.
    Grid points that land inside a wall are filtered away.
    Returns a single list containing the points.
    """
    nx, ny = np.shape(img)

    h = 100
    h_min = 1
    rho = signal.gaussian(n, n/4, sym=False)

    x_array = np.zeros(n)
    for i in range(1, n):
        h_next = max(h_min, h/rho[i])
        x_array[i] = x_array[i-1] + h_next
    y_array = np.copy(x_array)

    x_array = normalize(x_array, 0, nx-1)
    y_array = normalize(y_array, 0, ny-1)

    points = np.array([(i, j) for i in x_array for j in y_array])
    filtered_points = wall_filter(points, img)

    return filtered_points
