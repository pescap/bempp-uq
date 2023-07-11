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
from bemppUQ.shapes import kite, perturbate

ta = time.time()

bempp.api.global_parameters.assembly.potential_operator_assembly_type = "dense"
bempp.api.global_parameters.hmat.eps = 1e-4

config = bemppUQ.config.set_case("A")

parser = argparse.ArgumentParser(description="Set parameters")

parser.add_argument("--l0", default=0, type=int)

args = parser.parse_args()
l0 = args.l0

print(l0, "l0")

precision_list = [5, 10, 20]
precision = precision_list[l0]
h = 2.0 * np.pi / (precision * config["k_int"])
grid = kite(h=h)

print(grid.leaf_view.entity_count(1) * 2, "NDOF")

grid_eta, grid_fun = perturbate(grid, 0)

print("0")
transmission_operators = assemble_operators(grid, config)
print("1")
Umean, solution = evaluate_far_field(transmission_operators, config)
print("2")
result = evaluate_far_field_sd(
    grid, transmission_operators, config, solution, grid_fun, solve=True
)

ta = time.time() - ta

output = {}

output["time"] = ta
output["precision"] = precision
output["result"] = result
output["Umean"] = Umean

name = "results/full/" + str(precision) + ".txt"
np.save(name + ".npy", output)
