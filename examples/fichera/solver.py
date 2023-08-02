import argparse
import time

import bempp.api
import numpy as np

import bemppUQ
from bemppUQ.operators.maxwell import (
    assemble_operators,
    evaluate_far_field,
    evaluate_far_field_sd,
)

bempp.api.global_parameters.assembly.potential_operator_assembly_type = "dense"
bempp.api.global_parameters.hmat.eps = 1e-6
print(bempp.api.global_parameters.hmat.eps)

ta = time.time()


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


config = bemppUQ.config.set_case("A")

parser = argparse.ArgumentParser(description="Set parameters")

parser.add_argument("--l0", default=0, type=int)
parser.add_argument("--nrank", default=1, type=int)

parser.add_argument("--prec", default=10, type=int)

args = parser.parse_args()
l0 = args.l0
nrank = args.nrank
prec = args.prec

print(l0, "l0")


precision_list = [2, 5, 10]
precision = precision_list[l0]

if prec != 10:
    precision = prec

h = 2.0 * np.pi / (precision * config["k_int"])

if l0 == 2:
    grid = get_base_grid(h=h)
else:
    grid = bempp.api.shapes.reentrant_cube(h=h, refinement_factor=1)


print(grid.leaf_view.entity_count(1) * 2, "NDOF")

print("0")
transmission_operators = assemble_operators(grid, config)
print("1")
Umean, solution = evaluate_far_field(transmission_operators, config)
print("2")


grid_funs = []

for ii in range(nrank):
    for jj in range(nrank):
        print(ii, jj)

        def fun(point, n, domain_index, result):
            x, y, z = point

            res = 0j
            if z == 0.5 and (x <= 0.5) and (y <= 0.5):
                res += function(x, y, ii, jj)
            result[0] = res

        space = bempp.api.function_space(grid, "B-P", 1)
        grid_fun = bempp.api.GridFunction(space, fun=fun)
        grid_funs.append(grid_fun)


result = evaluate_far_field_sd(
    grid, transmission_operators, config, solution, grid_funs, solve=True
)

ta = time.time() - ta

output = {}

output["time"] = ta
output["precision"] = precision
output["result"] = result
output["Umean"] = Umean

name = "results/full/" + str(nrank) + "_" + str(precision) + ".txt"
np.save(name + ".npy", output)
