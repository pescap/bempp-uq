import argparse
import pickle
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
parser.add_argument("--l1", default=1, type=int)

args = parser.parse_args()
l0 = args.l0
l1 = args.l1

print(l0, l1, "l0", "l1")

precision_list = [5, 10, 20]
precision0 = precision_list[l0]
precision1 = precision_list[l1]

h0 = 2.0 * np.pi / (precision0 * config["k_int"])
h1 = 2.0 * np.pi / (precision1 * config["k_int"])

grid0 = kite(h=h0)
grid1 = kite(h=h1)

print(grid0.leaf_view.entity_count(1) * 2, "NDOF 1")
print(grid1.leaf_view.entity_count(1) * 2, "NDOF 0")

grid_eta0, grid_fun0 = perturbate(grid0, 0)
grid_eta1, grid_fun1 = perturbate(grid1, 0)

print("0")
transmission_operators0 = assemble_operators(grid0, config)
print("1")
transmission_operators1 = assemble_operators(grid1, config)
print("2")
Umean0, solution0 = evaluate_far_field(transmission_operators0, config)
print("3")
Umean1, solution1 = evaluate_far_field(transmission_operators1, config)
print("4")
result0 = evaluate_far_field_sd(
    grid0, transmission_operators0, config, solution0, grid_fun0, solve=True
)
print("5")
result1 = evaluate_far_field_sd(
    grid1, transmission_operators1, config, solution1, grid_fun1, solve=True
)

ta = time.time() - ta

output = {}

output["time"] = ta
output["precision0"] = precision0
output["precision1"] = precision1

output["result0"] = result0
output["result1"] = result1
output["Umean0"] = Umean0
output["Umean1"] = Umean1


with open(
    "results/full/" + str(precision0) + "-" + str(precision1) + "output_full.txt", "wb"
) as fp:
    pickle.dump(output, fp)
