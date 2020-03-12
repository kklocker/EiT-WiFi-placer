import matplotlib.pyplot as plt
import numpy as np
from dask import delayed, compute
import scipy

from wifi_matrix import generate_lu, solve_system, parse_image, generate_A
from score import step_score

from scipy.sparse import save_npz, load_npz
import os


def find_optimal_placement(lu, img, N=100):
    """
    Find the optimal placement of the router in an apartment.
    Assumes the lu-decomposition is found.
    N is the number of points to test at each run.
    """
    nx, ny = img.shape

    n = int(np.sqrt(N))
    n_new = n

    stepx = nx // n
    stepy = ny // n

    x = np.arange(0, nx + 1, stepx, dtype=np.uint16)
    y = np.arange(0, ny + 1, stepy, dtype=np.uint16)

    optimal_solution_found = False

    curr_best_idx = []
    curr_best_sol = []
    while not optimal_solution_found:
        scores = []
        solutions = []
        sol = np.array(solve_system(lu, x, y, img))

        for i in range(sol.shape[0]):
            score = delayed(step_score)((sol[i, :]).reshape(img.shape), img)
            scores.append(score)

        results = np.array(compute(*scores)).reshape((x.shape[0], y.shape[0]))
        tmp_idx = np.argmax(results)
        curr_best_sol.append(sol[tmp_idx, :])
        max_arg = np.unravel_index(tmp_idx, results.shape)

        x_new = x[max_arg[0]]
        y_new = y[max_arg[1]]
        n_new = n_new // 2

        nx = 1 if nx <= 1 else (nx // 2)
        ny = 1 if ny <= 1 else (ny // 2)
        curr_best_idx.append((x_new, y_new))

        if (nx == 1 & ny == 1) or (stepx == 1 & stepy == 1):
            optimal_solution_found = True
            break
        stepx = 1 if stepx <= 1 else (nx // n)
        stepy = 1 if stepy <= 1 else (ny // n)
        start_x = max(0, x_new - (nx // 2))
        stop_x = min(img.shape[0], x_new + (nx // 2))
        start_y = max(0, y_new - (ny // 2))
        stop_y = min(img.shape[1], y_new + (ny // 2))
        x = np.arange(start_x, stop_x, stepx, dtype=np.uint16)
        y = np.arange(start_y, stop_y, stepy, dtype=np.uint16)

    return curr_best_idx, curr_best_sol


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

