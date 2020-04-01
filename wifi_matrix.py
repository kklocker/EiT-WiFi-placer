#!/usr/bin/env python3
from math import floor
from dask import delayed, compute
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from itertools import zip_longest
import time


def subdivide_image(image, *, xboxes, yboxes):
    """
    Subdivides an image into xboxes*yboxes smaller boxes
    and yields the images
    """
    x_size, y_size = int(image.shape[0] / xboxes), int(image.shape[1] / yboxes)
    xdivs = [x * x_size for x in range(0, xboxes)]
    ydivs = [y * y_size for y in range(0, yboxes)]
    for x in xdivs:
        for y in ydivs:
            yield image[x : x + x_size, y : y + y_size]


def downsize_image(image, n):
    """
    Downsizes an image by selecting every n pixel, it might be possible to
    resize to an arbitary resolution, but I didn't want to deal with
    interpolation and strange behaviour with semi-transparent pixels.
    """
    return image[::n, ::n]


def parse_image(filename, n_air, n_concrete):
    """
    Imports a greyscale png image, and determines where the walls are from the greyscale value.
    Assuming concrete walls.
    """
    read_img = matplotlib.image.imread(filename)

    if len(np.shape(read_img)) > 2:  # In case image is not grayscale.
        read_img = read_img[:, :, 0]

    read_img = read_img.astype(np.complex64)
    read_img[read_img >= 0.9] = n_air
    read_img[read_img < 0.9] = n_concrete
    return read_img


def pad_image(img):
    """
    Surrounds the floorplan with absorbing material to stop reflections. pad_value should be massively complex
    to achieve this.
    """
    pad_width = 4  # Amount of pixels to pad with.
    pad_value = 1 - 100000j
    x, y = np.shape(img)

    padded_img = np.zeros((x + 2 * pad_width, y + 2 * pad_width)) + pad_value
    padded_img[pad_width : pad_width + x, pad_width : pad_width + y] = img

    return padded_img


def generate_lu(floor, *args):
    return scipy.sparse.linalg.splu(generate_A(floor, *args))


def generate_A(floor, k=2 * np.pi / 0.06, dx=0.01, dy=0.01):
    """
    Assumes floor is an array of complex refractive indexes.
    Returns A as a sparse csc matrix.
    """
    nx, ny = np.shape(floor)
    diag = np.zeros(nx * ny, dtype=np.complex64)
    for i in range(nx):
        for j in range(ny):
            diag[ny * i + j] = -2 / dx ** 2 - 2 / dy ** 2 + np.square(k / floor[i, j])

    A = scipy.sparse.diags(
        [1 / dy ** 2, 1 / dx ** 2, diag, 1 / dx ** 2, 1 / dy ** 2],
        [-ny, -1, 0, 1, ny],
        shape=(nx * ny, nx * ny),
        format="lil",
        dtype=np.complex64,
    )

    for m in range(1, nx):
        j = m * (ny)
        i = j - 1
        A[i, j], A[j, i] = 0, 0

    return A.tocsc()


def generate_A_higher_order(floor, k=2 * np.pi / 0.06, dx=0.01, dy=0.01):
    """
    Assumes floor is an array of complex refractive indexes.
    Returns A as a sparse csc matrix.
    """
    nx, ny = np.shape(floor)
    diag = np.zeros(nx * ny, dtype=np.complex64)

    for i in range(nx):
        for j in range(ny):
            diag[ny * i + j] = - 30 / (12*dx ** 2) - 30 / (12*dy ** 2) + np.square(k / floor[i, j])

    diag_x1 = 16 / (12*dx ** 2)
    diag_x2 = -1 / (12*dy ** 2)

    diag_y1 = 16 / (12*dy ** 2)
    diag_y2 = -1 / (12*dy ** 2)

    A = scipy.sparse.diags(
        [diag_y2, diag_y1, diag_x2, diag_x1, diag, diag_x1, diag_x2, diag_y1, diag_y2],
        [-2*ny, -ny, -2, -1, 0, 1, 2, ny, 2*ny],
        shape=(nx * ny, nx * ny),
        format="lil",
        dtype=np.complex64,
    )

    for m in range(1, nx):
        j = m * ny
        i = j - 1
        A[i, j], A[j, i] = 0, 0
        A[i-1, j], A[j, i-1] = 0, 0
        A[i, j+1], A[j+1, i] = 0, 0

    return A.tocsc()


def lu_solve(lu, b):
    """
    Helper function for dask parallelization.
    """
    return lu.solve(b)


def solve_system(lu, x, y, img):
    """
    Solves the system Ax = b given the LU decomposition of A.
    x and y are lists of coordinates for positions of source term.
    Returns an array of solutions, where [i] is solution for source coordinates [x[i], y[i]]
    """
    nx, ny = img.shape
    b = np.zeros((nx * ny, len(x) * len(y)), dtype=np.complex64)
    for i, (xi, yi) in enumerate([(i, j) for i in x for j in y]):
        # print(b.shape, ny, nx, xi, yi)
        b[
            ny * xi + yi, i
        ] = 1e3  # Place a singular source term in b. Not sure what value it should be.

    sol = []
    for i in range(np.size(b, 1)):
        new = delayed(lu_solve)(lu, b[:, i])
        sol.append(new)
    sol = compute(*sol)

    return sol


def solve_single_system(lu, x, y, img_shape):
    """
    Solves system for a single point. To be parallellized. 
    """
    nx, ny = img_shape
    b = np.zeros(nx * ny, dtype=np.complex64)
    b[ny * x + y] = 1.0
    return lu.solve(b)


def plot_solution(x, img, n_concrete):
    """
     Reshapes x to 2D and plots the result.
     lower and upper set the color scaling.
    """
    nx, ny = img.shape
    x = x.reshape(nx, ny)
    x = 10 * np.log10(np.absolute(x) ** 2)  # Need a reference value for dB scale?
    x = np.ma.array(
        x, mask=img == n_concrete
    )  # Masks walls so the field is not plotted there. Optional.
    plt.figure(figsize=(ny / 100, nx / 100))
    plt.gca().patch.set_color("0.0")
    plt.contourf(
        x,
        80,
        corner_mask=False,
        cmap="jet",
        origin="lower",
        vmin=-50,
        vmax=-28,
        extend="both",
    )
    # plt.show()


if __name__ == "__main__":

    # Constants
    wavelength = 0.06  # Wavelength of WiFi in meters: 0.12 for 2.5GHz; 0.06 for 5GHz.
    k = 2 * np.pi / wavelength
    n_air = 1
    n_concrete = 2.16 - 0.021j  # Should depend on wavenumber.

    # Image and grid
    img = parse_image("plan-1k.png", n_air, n_concrete)
    print("Image size: ", np.shape(img))
    # L = 10
    nx, ny = np.shape(img)
    dx = 0.01
    dy = dx
    print(
        "Each pixel is ",
        round(dx * 100, 2),
        "cm, or ",
        round(100 * dx / wavelength, 2),
        "% of the wavelength.",
        sep="",
    )

    # Generate matrix A and the LU decomposition. It is unique for a given floorplan, stepsize, and frequency.
    tic = time.time()
    A = generate_A(img, k)
    LU = scipy.sparse.linalg.splu(A)
    toc = time.time()
    print("A and LU decomposition time: ", round(toc - tic, 3))

    # Coordinates for placement of WiFi source. These are passed to the solver.
    x_coord = [150, 640, 650]
    y_coord = [500]

    # Solve the system
    tic = time.time()

    sol = solve_system(LU, x_coord, y_coord, img)
    toc = time.time()
    print("Solve time: ", toc - tic)
    print("Time per position: ", (toc - tic) / np.size(sol, 0))
    print(np.shape(sol))

    for i in range(np.size(sol, 0)):
        plot_solution(sol[i], img, n_concrete)
    plt.show()
