import glob
import random
import time

import bempp.api
import numpy as np

import bemppUQ
from bemppUQ.operators.maxwell import assemble_operators, evaluate_far_field

bempp.api.global_parameters.assembly.potential_operator_assembly_type = "dense"
bempp.api.global_parameters.hmat.eps = 1e-4

# Get a seed
t0 = time.time()

nrank = 6

text = glob.glob("results/*.txt")
seed = random.randint(0, 10000)

print("Starting the simulation for seed: ", seed)


def function(x, y, i, j):
    zx = 0
    zy = 0
    if i == 0:
        zx = np.sin(x * np.pi * 2)
    if i == 1:
        if x <= 0.25:
            zx = np.sin(x * np.pi * 4)
    if i == 2:
        if x > 0.25:
            zx = -np.sin(x * np.pi * 4)
    if i == 3:
        if x <= 0.5 / 3.0:
            zx = np.sin(x * np.pi * 6)
    if i == 4:
        if x > 0.5 / 3.0 and x <= 1 / 3.0:
            zx = -np.sin(x * np.pi * 6)
    if i == 5:
        if x > 1 / 3.0:
            zx = np.sin(x * np.pi * 6)

    if j == 0:
        zy = np.sin(y * np.pi * 2)

    if j == 1:
        if y <= 0.25:
            zy = np.sin(y * np.pi * 4)

    if j == 2:
        if y > 0.25:
            zy = -np.sin(y * np.pi * 4)

    if j == 3:
        if y <= 0.5 / 3.0:
            zy = np.sin(y * np.pi * 6)

    if j == 4:
        if y > 0.5 / 3.0 and y <= 1 / 3.0:
            zy = -np.sin(y * np.pi * 6)
    if j == 5:
        if y > 1 / 3.0:
            zy = np.sin(y * np.pi * 6)

    return zx * zy


# Def random:


def Phiz(point, n, domain_index, result):
    x, y, z = point

    res = 0j
    if z == 0.5 and (x <= 0.5) and (y <= 0.5):
        for ii in range(nrank):
            for jj in range(nrank):
                res += Random[ii, jj] * function(x, y, ii, jj)
    result[0] = res


def perturbate(grid, t, kappa_pert=None):
    P1 = bempp.api.function_space(grid, "B-P", 1)
    grid_funz = bempp.api.GridFunction(P1, fun=Phiz)
    elements = grid.leaf_view.elements
    vertices = grid.leaf_view.vertices
    x, y, z = vertices

    vertices[2, :] = z + t * grid_funz.coefficients
    return bempp.api.grid_from_element_data(vertices, elements)


def get_base_grid(h):
    gr0 = bempp.api.shapes.reentrant_cube(h=h, refinement_factor=1)
    elements0 = list(gr0.leaf_view.entity_iterator(0))
    N0 = len(elements0)
    tol = h / 10.0
    for i in range(N0):
        el0 = elements0[i]
        z = el0.geometry.corners[2]
        if np.linalg.norm(np.array([1 / 2, 1 / 2, 1 / 2]) - z) < tol:
            gr0.mark(el0)
    gr1 = gr0.refine()

    elements0 = list(gr1.leaf_view.entity_iterator(0))
    N0 = len(elements0)
    for i in range(N0):
        el0 = elements0[i]
        z = el0.geometry.corners[2]
        if np.linalg.norm(np.array([1 / 2, 1 / 2, 1 / 2]) - z) < tol:
            gr1.mark(el0)

    base_grid = gr1.refine()
    return base_grid


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


t = 0.05
set_random_seed(seed)

config = bemppUQ.config.set_case("MC")

precision = 30

h = 2.0 * np.pi / (precision * config["k_int"])

base_grid = get_base_grid(h=h)
print(base_grid.leaf_view.entity_count(1) * 2, "NDOF")


Random = np.random.rand(6, 6) * 2 - 1
gridt = perturbate(base_grid, t)

transmission_operators_t = assemble_operators(gridt, config)

Ut, _ = evaluate_far_field(transmission_operators_t, config)

np.savetxt("results1/out" + str(seed) + ".txt", Ut.view(float))

t0 = time.time() - t0
print("Execution time: ", t0)
