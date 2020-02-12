import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import time


def generate_floor(n):
    """
    Generates a floorplan, with value 0 on the interior and 1 on boundaries.
    """
    floor = np.zeros((n,n))
    floor[0,:] = 1
    floor[-1, :] = 1
    floor[:, 0] = 1
    floor[:, -1] = 1
    floor[round(n/4), 0:round(n/2)+1] = 1
    floor[0:round(n/4), round(n/2)] = 1
    return floor


def generate_Ab(floor):
    """
    Generates matrices A and b. Returns A as a sparse matrix.
    """
    dx2 = (L / N) ** 2

    nu = 1  # Refractive index. Should depend on position
    k = 2*np.pi/0.06   # Wavenumber

    nx = N - 1
    ny = N - 1
    n = (nx) * (ny)  # number of unknowns

    d = np.ones(n)  # diagonals
    b = np.zeros(n)  # RHS
    d0 = d * (-2/dx2 - 2/dx2 + (k**2)/(nu**2))
    d1 = d[0:-1]/dx2
    d5 = d[0:-ny]/dx2

    A = scipy.sparse.diags([d0, d1, d1, d5, d5], [0, 1, -1, ny, -ny], format='csc')

    # set elements to zero in A matrix where BC are imposed
    for k in range(1, nx):
        j = k * (ny)
        i = j - 1
        A[i, j], A[j, i] = 0, 0
        b[i] = -Ttop

    # Boundary values:
    b[-ny:] += -Tright  # set the last ny elements to -Tright
    b[-1] += -Ttop  # set the last element to -Ttop
    b[0:ny - 1] += -Tleft  # set the first ny elements to -Tleft
    b[0::ny] += -Tbottom  # set every ny-th element to -Tbottom

    # Singular point value:
    b[ny*20] = 1  # Place a singular source term in b

    return A, b


def plot_solution(x):
    """
     Reshapes 1D x to 2D and plots the result.
    """
    x = x.reshape((N-1, N-1))
    plt.figure()
    plt.contourf(x, 100, corner_mask = False, cmap = "jet")
    plt.show()


L = 1   # Side length of (square) floor.
N = 150 # Grid points.

# Set values at the boundaries
Ttop = 0
Tbottom = 0
Tleft = 0
Tright = 0


floor = generate_floor(N)
A, b = generate_Ab(floor)
x = scipy.sparse.linalg.spsolve(A, b)
plot_solution((x)**2)

"""
TODO:   Change elements of A to also consider interior walls.
        Figure out value of source term.
"""