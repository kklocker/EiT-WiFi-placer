import matplotlib.pyplot as plt
import numpy as np
from dask import delayed, compute

from wifi_matrix import generate_lu, solve_system, parse_image
from score import step_score


def find_optimal_placement(lu, img, N=100):
    """
    Find the optimal placement of the router in an apartment.
    Assumes the lu-decomposition is found.
    N is the number of points to test at each run.
    """
    nx, ny = img.shape

    n = int(np.sqrt(N))

    stepx = n // nx
    stepy = n // ny
    x = np.arange(0, nx, stepx, dtype=np.uint16)
    y = np.arange(0, ny, stepx, dtype=np.uint16)

    optimal_solution_found = False

    while not optimal_solution_found:
        scores = []

        for i in range(len(x)):
            for j in range(len(y)):
                sol = delayed(solve_system)(lu, [x[i]], [y[j]])
                # idx = i*n + j
                score = step_score(sol[:, 0], img)
                scores.append(score)

        results = np.array(compute(scores)).reshape((n, n))
        max_arg = np.argmax(results)

        x_new = x[max_arg[0]]
        y_new = y[max_arg[1]]
        n_new = n // 2

        if n_new == 1:
            optimal_solution_found = True

        n_new_x = min(n_new, (x[-1] - x[0]))
        n_new_y = min(n_new, (y[-1] - y[0]))

        start_x = max(0, x_new - n_new_x // 2)
        stop_x = min(nx, x_new + n_new_x // 2)

        start_y = max(0, y_new - n_new_y // 2)
        stop_y = min(ny, y_new + n_new_y // 2)

        new_step_x = n // (stop_x - start_x)
        new_step_y = n // (stop_y - start_y)

        x = np.arange(start_x, stop_x, new_step_x, dtype=np.uint16)
        y = np.arange(start_y, stop_y, new_step_y, dtype=np.uint16)

    return x_new, y_new


if __name__ == "__main__":

    wavelength = 0.06  # Wavelength of WiFi in meters: 0.12 for 2.5GHz; 0.06 for 5GHz.
    k = 2 * np.pi / wavelength
    n_air = 1
    n_concrete = 2.16 - 0.021j  # Should depend on wavenumber.
    img = parse_image("plan-1k.png", n_air, n_concrete)
    lu = generate_lu(img, k)

    x, y = find_optimal_placement(lu, img)

