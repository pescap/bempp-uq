import glob
import random

# Get a seed
import time

import bempp.api
import numpy as np

import bemppUQ
from bemppUQ.operators.maxwell import assemble_operators, evaluate_far_field
from bemppUQ.shapes import kite, perturbate

t0 = time.time()
text = glob.glob("results/*.txt")

# S = [re.sub("[^0-9]", "",text) for text in text]
# S = np.array(S, dtype=int)
# seed = next(iter(set(range(min(S)+1, max(S))) - set(S)))

bempp.api.global_parameters.assembly.potential_operator_assembly_type = "dense"
bempp.api.global_parameters.hmat.eps = 1e-4


seed = random.randint(0, 10000)

print("Starting the simulation for seed: ", seed)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


t = 0.05
set_random_seed(seed)

config = bemppUQ.config.set_case("MC")
precision = 20

h = 2.0 * np.pi / (precision * config["k_int"])

grid = kite(h=h)
print(grid.leaf_view.entity_count(1) * 2, "NDOF")


Random = np.random.rand(1, 1) * 2 - 1
gridt, grid_fun = perturbate(grid, Random * t)

transmission_operators_t = assemble_operators(gridt, config)

Ut, _ = evaluate_far_field(transmission_operators_t, config)

np.savetxt("results/out" + str(seed) + ".txt", Ut.view(float))

t0 = time.time() - t0
print("Execution time: ", t0)
