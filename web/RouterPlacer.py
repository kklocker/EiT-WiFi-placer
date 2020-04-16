import numpy as np
from scipy.sparse.linalg import splu
from time import time
import matplotlib.pyplot as plt

from wifi_matrix import (
    parse_image_file,
    pad_image,
    generate_A_higher_order,
    solve_single_system,
)
from find_optimal import find_optimal_2, convolve_solution
from score import step_score


class RouterPlacer:
    def __init__(
        self,
        floorplan_image,
        wavelength=0.06,
        n_air=1 + 0j,
        n_material=2.5 - 1.0j * (2.5 * 0.2),
        convolve=True,
    ):
        self.k = 2 * np.pi / wavelength
        self.n_air = n_air
        self.n_material = n_material
        self.convolve = convolve

        tmp_img = parse_image_file(floorplan_image, n_air, n_material)
        tmp_img = pad_image(tmp_img)
        self.img = tmp_img

        self.x = None
        self.y = None
        self.lu = None
        self.optimal_coords_found = False
        self.scores = None
        self.optimal_score = None
        self.optimal_solution = None

    def get_lu(self):
        if self.lu is not None:
            return
        print("Setting up matrix.")
        a = time()
        A = generate_A_higher_order(self.img, self.k)
        b = time()
        print(f"Setting up matrix took {b-a:.2f}s")
        print("Fetching LU-decomposition.")
        lu = splu(A)
        c = time()
        print(f"LU-decomposition too {c-b:.2f}s")
        self.lu = lu

    def get_results(self):
        if self.lu is None:
            print("LU-decomposition not found. Getting new.")
            self.get_lu()

        x, y, scores, _ = find_optimal_2(
            self.lu, self.img, N=21, convolve=self.convolve
        )

        self.x, self.y = x, y
        self.optimal_coords_found = True
        self.scores = scores

    def get_optimal_solution(self):
        if not self.optimal_coords_found:
            print("Optimal coordinates not found. Getting new results")
            self.get_results()

        sol = solve_single_system(self.lu, self.x, self.y, self.img.shape)
        if self.convolve:
            sol = convolve_solution(sol, self.img.shape)

        self.optimal_solution = sol
        self.optimal_score = step_score(sol, self.img)

    def score_plot(self):
        plt.imshow(self.scores)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title("Scores")
        plt.savefig("web/scores.png")
        plt.show()

    def solution_plot(self):
        plt.figure(figsize=(14, 10))
        plt.imshow(
            np.ma.array(
                10
                * np.log10(
                    np.abs(self.optimal_solution.reshape(self.img.shape)) ** 2
                    / (np.max(np.abs(self.optimal_solution) ** 2))
                ),
                mask=(self.img != 1.0),
            ),
            cmap="jet",
        )
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title("Optimal placement.", size=20)
        plt.scatter(self.y, self.x)
        plt.savefig("web/optimal_placement.png")
        plt.show()
