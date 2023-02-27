#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../../')
import bemppUQ
import numpy as np
import bempp.api 
from bemppUQ.shapes import kite, perturbate
from bemppUQ.functions import tangential_trace, neumann_trace
from bemppUQ.utils.login import rescale
from bemppUQ.operators.maxwell import multitrace_identity, multitrace_operator, assemble_operators, evaluate_far_field, evaluate_far_field_sd

number_of_angles = 400
angles = np.pi * np.linspace(0, 2, number_of_angles)
far_field_points = np.array([np.cos(angles), np.sin(angles), np.zeros(number_of_angles)])
case = 'LF'

if case == 'LF':
    polarization = np.array([1.0 + 1j, 2.0, -1.0 - 1.0 / 3.0 * 1j])
    direction = np.array([1.0 / np.sqrt(14), 2.0 / np.sqrt(14), 3.0 / np.sqrt(14)], dtype='float64')

    eps_rel = 1.9
    mu_rel = 1.

    k_ext = 1.047197551196598
    k_int = k_ext * np.sqrt(eps_rel * mu_rel)

print("The exterior wavenumber is: {0}".format(k_ext))
print("The interior wavenumber is: {0}".format(k_int))

precision = 30
h = 2.0 * np.pi / (precision * k_int)

t_list = [0.05,0.1,0.25,0.5,1]

base_grid = kite(h=h) 
grid_eta, grid_fun = perturbate(base_grid, 0)

# far_field for t=0
transmission_operators = assemble_operators(base_grid, k_int, k_ext, far_field_points)
far_field, solution = evaluate_far_field(transmission_operators, eps_rel, mu_rel, k_ext, polarization, direction)

# far_field for the SD
far_field_p = evaluate_far_field_sd(base_grid, transmission_operators, eps_rel, mu_rel, k_int, k_ext, polarization, direction, solution, grid_fun)

far_field_eta_list = []

for t in t_list:
    print(t, 't now')
    grid_eta, _ = perturbate(base_grid, t)
    transmission_operators_eta = assemble_operators(grid_eta, k_int, k_ext, far_field_points)
    far_field_eta, _ = evaluate_far_field(transmission_operators_eta, eps_rel, mu_rel, k_ext, polarization, direction)
    
    far_field_eta_list.append(far_field_eta)
    
    residual_eta = far_field_eta - far_field
    residual_p = far_field_eta - far_field - t * far_field_p
    
    err_0 = np.linalg.norm(residual_eta)/np.linalg.norm(far_field_eta)
    err_FOA = np.linalg.norm(residual_p)/ np.linalg.norm(far_field_eta)
    
    print(err_0, 'relative norm for the residual eta')
    print(err_FOA, 'norm for the residual first order')
    
    
name = (
    "results/"
        + "precision"
    + str(precision)
    + "_case"
    + str(case))


my_dict = {
        "far_field": far_field,
    "far_field_p": far_field_p,
    "far_field_eta": far_field_eta_list,
    }

np.save(name + ".npy", my_dict)    