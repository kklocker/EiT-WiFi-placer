import numpy as np
from numba import jit


def create_gaussian(N = 1000, sigma=1.0, mu=0):
    """
    Kun for testing
    """
    x, y = np.meshgrid(np.linspace(-1,1,N), np.linspace(-1,1,N))
    d = np.sqrt(x*x+y*y)
    return 1./(sigma * np.sqrt(2*np.pi))*np.exp(-((d-mu)**2 / ( 2.0 * sigma**2 ) ) )


@jit
def basic_score(sol, img):
    """
    Finn ut hva som skjer utenfor boundary.
    """
    # u = np.abs(u)
    u = np.ma.array(np.abs(sol).reshape(img.shape), mask = (img !=1.))
    area = u.count()
    return np.sum(u) / area


@jit
def step_score(sol, img):
    """
    Minimum signal: u0
    """

    umax = np.max(sol)
    db = 10*np.log10(sol/umax)    
    
    A = np.ma.array(np.abs(sol).reshape(img.shape), mask = (img !=1.))
    A = u.shape[0]*u.shape[1]
    print(f"DECIBEL: {db}")
    #tmp = np.where(u>=u0, u, 0)
    tmp = umax[db> -70]
    return np.sum(tmp) / A

#NB: mye tregere.
@jit
def weighted_score(u, p, degree = 1.):
    """
    p: position of source (tuple of indices)
    u: solution of wave-eq.
    """
    temp = np.zeros_like(u)
    A = u.shape[0]*u.shape[1]
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            r = np.linalg.norm([i-p[0], j-p[1]])
            temp[i,j] = (r**degree)*u[i,j]
    return np.sum(temp) / A
    