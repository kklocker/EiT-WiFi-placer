import matplotlib.pyplot as plt
import numpy as np
from dask import Delayed, Compute

from wifi_matrix import generate_A, solve_system
from score import step_score



def find_optimal_placement(lu, img, N = 100):
    """
    Find the optimal placement of the router in an apartment.
    Assumes the lu-decomposition is found.
    N is the number of points to test at each run.
    """
    nx, ny = img.shape

    n = int(np.sqrt(N))
        

    x = np.linspace(0,nx, n, dtype=np.uint16)
    y = np.linspace(0,ny, n, dtype=np.uint16)
    
    optimal_solution_found = False


    
    while not optimal_solution_found:
        scores = []

        for i in range(n):
            for j in range(n):
                sol = Delayed(solve_system)(lu, [x[i]], [y[j]])
                #idx = i*n + j
                score = step_score(sol[:,0],img)
                scores.append(score)

        results = np.array(Compute(scores)).reshape((n,n))
        max_arg = np.argmax(results)

        x_new = x[max_arg[0]]
        y_new = y[max_arg[1]]

        n_new = n//2

        if n_new ==1:
            optimal_solution_found = True

        x = np.linspace([start], [stop], [num])
