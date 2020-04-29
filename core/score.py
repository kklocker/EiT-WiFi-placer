import numpy as np
from numba import jit
from dask.array import ma


def create_gaussian(N=1000, sigma=1.0, mu=0):
    """
    Kun for testing
    """
    x, y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    d = np.sqrt(x * x + y * y)
    return (
        1.0
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    )


@jit
def basic_score(sol, img):
    """
    Finn ut hva som skjer utenfor boundary.
    """
    # u = np.abs(u)
    u = np.ma.array(np.abs(sol).reshape(img.shape), mask=(img != 1.0))
    area = u.count()
    return np.sum(u) / area


@jit(nopython=False, forceobj=True)
def step_score(sol, img, threshold=-50):
    """
    Minimum signal: u0
    """

    # umax = 1e3  # np.max(sol)
    umax = np.max(np.square(np.abs(sol)))
    db = 10 * np.log10(np.square(np.abs(sol)) / umax).reshape(img.shape)
    # A = ma.masked_array(np.abs(sol).reshape(img.shape), mask=(img != 1.0))
    A = np.ma.array(np.square(np.abs(sol)).reshape(img.shape), mask=(img != 1.0))
    # A = np.ma.array(np.ones_like(img), mask=(img != 1.0))
    area = A.count()

    tmp = A[db > threshold]
    return np.sum(tmp) / area


# NB: mye tregere.
@jit
def weighted_score(u, p, degree=1.0):
    """
    p: position of source (tuple of indices)
    u: solution of wave-eq.
    """
    temp = np.zeros_like(u)
    A = u.shape[0] * u.shape[1]
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            r = np.linalg.norm([i - p[0], j - p[1]])
            temp[i, j] = (r ** degree) * u[i, j]
    return np.sum(temp) / A
