import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from itertools import zip_longest
import time


def parse_image(filename):
    """
    Imports a greyscale png image, and determines where the walls are from the greyscale value.
    Assuming concrete walls.
    """
    read_img = matplotlib.image.imread(filename)

    if len(np.shape(read_img)) > 2:     # In case image is not grayscale.
        read_img = read_img[:,:,0]

    read_img = read_img.astype(np.complex64)
    read_img[read_img >= 0.9] = n_air
    read_img[read_img < 0.9] = n_concrete
    return read_img


def generate_A(floor, k = 2*np.pi/0.06):
    """
    Assumes floor is an array of complex refractive indexes.
    Returns A as a sparse csc matrix.
    """
    nx, ny = np.shape(img)
    diag = np.zeros(nx*ny, dtype=np.complex64)
    for i in range(nx):
        for j in range(ny):
            diag[ny*i + j] = -2/dx**2 - 2/dy**2 + np.square(k/floor[i, j])

    A = scipy.sparse.diags([1/dy**2, 1/dx**2, diag, 1/dx**2, 1/dy**2], [-ny, -1, 0, 1, ny], shape=(nx*ny, nx*ny), format='lil', dtype = np.complex64)

    for m in range(1, nx):
        j = m * (ny)
        i = j - 1
        A[i, j], A[j, i] = 0,0

    return A.tocsc()


def solve_system(lu, x, y):
    """
    Solves the system Ax = b given the LU decomposition of A.

    x and y are lists of coordinates for positions of source term.

    Returns an array of solutions, where [:, i] is solution for source coordinates [x[i], y[i]]
    """
    b = np.zeros((nx * ny, max(np.size(x), np.size(y))), dtype=np.complex64)
    for i, (xi, yi) in enumerate(zip_longest(x, y, fillvalue = np.where(np.size(x) < np.size(y), x[-1], y[-1]))):
        b[ny * yi + xi, i] = 1e3   # Place a singular source term in b. Not sure what value it should be.

    return lu.solve(b)


def plot_solution(x):
    """
     Reshapes x to 2D and plots the result.
     lower and upper set the color scaling.
    """
    x=x.reshape(nx,ny)
    x = 10*np.log10(np.absolute(x)**2) # Need a reference value for dB scale?
    x = np.ma.array(x, mask = img == n_concrete) # Masks walls so the field is not plotted there. Optional.
    plt.figure(figsize=(ny/100, nx/100))
    plt.gca().patch.set_color('0.2')
    plt.contourf(x, 80, corner_mask = False, cmap = "jet", origin='lower', vmin = -50, vmax = -28, extend = 'both')
    #plt.show()


if __name__ == '__main__':

    # Constants
    wavelength = 0.06     # Wavelength of WiFi in meters: 0.12 for 2.5GHz; 0.06 for 5GHz.
    k = 2*np.pi/wavelength
    n_air = 1
    n_concrete = 2.16 - 0.021j     # Should depend on wavenumber.

    # Image and grid
    img = parse_image('plan-1k.png')
    print("Image size: ", np.shape(img))
    #L = 10
    nx, ny = np.shape(img)
    dx = 0.01
    dy = dx
    print("Each pixel is ", round(dx*100, 2), "cm, or ", round(100*dx/wavelength,2), "% of the wavelength.", sep = "")

    # Generate matrix A and the LU decomposition. It is unique for a given floorplan, stepsize, and frequency.
    tic = time.time()
    A = generate_A(img, k)
    LU = scipy.sparse.linalg.splu(A)
    toc = time.time()
    print("A and LU decomposition time: ", round(toc - tic, 3))

    # Coordinates for placement of WiFi source. These are passed to the solver.
    x_coord = [55, 100, 150, 200, 250,  300, 350, 400, 450]
    y_coord = [350]

    # Solve the system
    tic = time.time()
    sol = solve_system(LU, x_coord, y_coord)
    toc = time.time()
    print("Solve time: ", round(toc-tic, 3))
    print("Time per position: ", round((toc-tic)/9, 3))


    for i in range(np.size(sol, 1)):
        plot_solution(sol[:,i])
    plt.show()
