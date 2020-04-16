import matplotlib.pyplot as plt
import numpy as np
from dask import delayed, compute
import dask.array as da
import scipy
from scipy.signal import convolve2d

from wifi_matrix import (
    generate_lu,
    solve_system,
    parse_image,
    generate_A,
    solve_single_system,
)
from gaussianGrid import gauss_grid
from score import step_score

from scipy.sparse import save_npz, load_npz
import os
from time import time


def find_optimal_placement(lu, img, N=100):
    """
    Find the optimal placement of the router in an apartment.
    Assumes the lu-decomposition is found.
    N is the number of points to test at each run.
    """
    nx, ny = img.shape

    n = int(np.sqrt(N))
    n_new = n
    #     print(n)

    x = np.linspace(0, nx, num=n + 2, endpoint=True, dtype=int)[1:-1]
    y = np.linspace(0, ny, num=n + 2, endpoint=True, dtype=int)[1:-1]

    stepx = x[1] - x[0]  # nx // (n)
    stepy = y[1] - y[0]  # ny // (n)
    optimal_solution_found = False
    all_tested_points = []
    curr_best_idx = []
    # curr_best_sol = []
    while not optimal_solution_found:
        all_tested_points.append([x, y])
        results = scores_from_point_lists(lu, img, x, y)
        tmp_idx = np.argmax(results)
        max_arg = np.unravel_index(tmp_idx, results.shape)
        x_new = x[max_arg[0]]
        y_new = y[max_arg[1]]
        n_new = n_new // 2
        nx = 1 if nx <= 1 else (nx // 2)
        ny = 1 if ny <= 1 else (ny // 2)
        curr_best_idx.append((x_new, y_new))
        # print(f"new (x, y): ({x_new}, {y_new}). \t Score: {results[max_arg]}")

        if (nx == 1 & ny == 1) or (stepx == 1 & stepy == 1):
            optimal_solution_found = True
            break

        stepx = 1 if stepx <= 1 else (nx // n)
        stepy = 1 if stepy <= 1 else (ny // n)

        start_x = max(0, x_new - (nx // 2))
        stop_x = min(img.shape[0], x_new + (nx // 2))
        start_y = max(0, y_new - (ny // 2))
        stop_y = min(img.shape[1], y_new + (ny // 2))

        print((start_x - stop_x), (start_y - stop_y))

        if (abs(start_x - stop_x) < 25) and (abs(start_y - stop_y) < 25):
            optimal_solution_found = True
            x_new = abs(start_x - stop_x) // 2
            y_new = abs(start_y - stop_y) // 2
            curr_best_idx.append((x_new, y_new))
            break

        x = np.arange(start_x, stop_x, stepx, dtype=np.uint16)
        y = np.arange(start_y, stop_y, stepy, dtype=np.uint16)

        x = np.append(x, x_new)
        y = np.append(y, y_new)

    all_tested_points = np.array(all_tested_points)
    return curr_best_idx, all_tested_points  # , curr_best_sol


def find_optimal_2(lu, img, N=20, convolve=True):

    # get position list
    #     x, y = get_position_list(img.shape, N)
    xx, yy = zip(*gauss_grid(img, n=N))
    x = np.unique(xx)
    y = np.unique(yy)
    print(f"Got gauss-distributed points. Now checking {x.shape[0]*y.shape[0]} points")
    start = time()
    scores = scores_from_point_lists(lu, img, x, y, convolve=convolve)
    end = time()

    print(f"Getting solutions took {end-start:.2f}s")

    tmp_idx = np.argmax(scores)
    # print(tmp_idx)
    max_arg = np.unravel_index(tmp_idx, scores.shape)
    # print(f"{max_arg})

    x_best, y_best = x[max_arg[0]], y[max_arg[1]]
    print(f"Optimal position: {(x_best, y_best)}")

    return x_best, y_best, scores, max_arg


def scores_from_point_lists(lu, img, x, y, convolve=True):
    """
    Returns a list of scores from x-and y-positions.
    Assumes the image has been padded with absorpion at infinity.
    """

    scores = []
    # solutions = []
    for i, (xi, yi) in enumerate([(i, j) for i in x for j in y]):
        sol = delayed(solve_single_system)(lu, xi, yi, img.shape)
        if convolve:
            sol = delayed(convolve_solution)(sol, img.shape, (3, 3))
        score = delayed(step_score)(sol, img)
        scores.append(score)
    # print(len(scores), len(x), len(y))
    results = np.array(compute(*scores)).reshape(
        x.shape[0], y.shape[0]
    )  # (9,622,1000)    return results
    return results


def convolve_solution(sol, img_shape, conv_shape=(3, 3)):
    """TODO: Sjekk om dette er en fornuftig ting å gjøre. Også om det skal gjøres før/etter abs^2.
    
    Arguments:
        sol {[type]} -- [description]
        size {[type]} -- [description]
    """

    i1, i2 = conv_shape
    n = i1 * i2
    kernel = np.ones((i1, i2)) / n
    convolved = convolve2d(sol.reshape(img_shape), kernel, mode="same").reshape(
        sol.shape
    )
    return convolved


if __name__ == "__main__":

    wavelength = 0.06  # Wavelength of WiFi in meters: 0.12 for 2.5GHz; 0.06 for 5GHz.
    k = 2 * np.pi / wavelength
    n_air = 1
    n_concrete = 2.16 - 0.021j  # Should depend on wavenumber.
    img = parse_image("plan-1k.png", n_air, n_concrete)

    if not os.path.exists("plan-1k-A.npz"):
        print("Generating new A. ")
        A = generate_A(img, k)
        save_npz("plan-1k-A", A)
    else:
        print("Loading A.")
        A = load_npz("plan-1k-A.npz")

    lu = scipy.sparse.linalg.splu(A)

    x, y = find_optimal_placement(lu, img)
