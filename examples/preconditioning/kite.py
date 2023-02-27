#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../../')
import bemppUQ
import numpy as np
import bempp.api 
from bemppUQ.shapes import kite, perturbate
from bemppUQ.functions import tangential_trace, neumann_trace
from bemppUQ.utils.login import rescale, gmres
from bemppUQ.operators.maxwell import multitrace_identity, multitrace_operator, assemble_operators, evaluate_far_field, evaluate_far_field_sd
import time 

from bemppUQ.assembly.operators import DenseTensorLinearOperator

from bempp.api.assembly.blocked_operator import \
        coefficients_of_grid_function_list, \
        projections_of_grid_function_list, \
        grid_function_list_from_coefficients
import argparse

parser = argparse.ArgumentParser(description="Case")
parser.add_argument("--setting", default='u', type=str)
parser.add_argument("--maxiter", default=20, type=int)

parser.add_argument("--MtEType", default='1', type=str)

parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="save results",
)

ta = time.time()

args = parser.parse_args()

setting = args.setting
MtEType = args.MtEType
maxiter = args.maxiter
save = args.save

restart = maxiter
tolerance = 1e-4

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
base_grid = kite(h=h) 


print(base_grid.leaf_view.entity_count(1) * 2, 'Ndof')


if setting in ['o', 'O']:
    transmission_operators = assemble_operators(base_grid, k_int, k_ext, far_field_points, spaces='maxwell_primal', osrc=True, MtEType=MtEType)

    multitrace_int, multitrace_ext, osrc_int, osrc_ext, identity, electric_far, magnetic_far = transmission_operators

    rescaled_osrc_int_op = rescale(osrc_int, np.sqrt(eps_rel), np.sqrt(mu_rel))
    rescaled_int_op = rescale(multitrace_int, np.sqrt(eps_rel), np.sqrt(mu_rel))

    lhs_osrc = rescaled_osrc_int_op +osrc_ext
    lhs_op = rescaled_int_op + multitrace_ext
    rhs_op = .5 * identity - rescaled_int_op
    electric_incident = bempp.api.GridFunction(lhs_op.domain_spaces[0], fun=tangential_trace(k_ext, polarization, direction))
    magnetic_incident = bempp.api.GridFunction(lhs_op.domain_spaces[1], fun=neumann_trace(k_ext, polarization, direction))

    rhs = rhs_op * [electric_incident, magnetic_incident]
    b = projections_of_grid_function_list(rhs, lhs_op.dual_to_range_spaces)
    lhs_op_wf = lhs_op.weak_form()
    P = lhs_osrc.weak_form()
    A = P * lhs_op_wf
    f = P * b
        
    if setting == 'o':
        print('solver osrc')
        x, info, res, timing = gmres(A, f, return_residuals=True, restart=restart, maxiter=maxiter, tol=tolerance)
    else:
        AA = DenseTensorLinearOperator(A, A)
        ff = np.dot(np.array([f]).T, np.array([f])).ravel()
        print('solver Osrc')
        x, info, res, timing = gmres( AA ,ff, return_residuals=True,restart=restart, maxiter=maxiter, tol=tolerance)

elif setting in ['u', 'U']:
    transmission_operators = assemble_operators(base_grid, k_int, k_ext, far_field_points, spaces='maxwell_primal', osrc=True)

    multitrace_int, multitrace_ext, osrc_int, osrc_ext, identity, electric_far, magnetic_far = transmission_operators

    rescaled_osrc_int_op = rescale(osrc_int, np.sqrt(eps_rel), np.sqrt(mu_rel))
    rescaled_int_op = rescale(multitrace_int, np.sqrt(eps_rel), np.sqrt(mu_rel))

    lhs_osrc = rescaled_osrc_int_op +osrc_ext
    lhs_op = rescaled_int_op + multitrace_ext
    rhs_op = .5 * identity - rescaled_int_op
    electric_incident = bempp.api.GridFunction(lhs_op.domain_spaces[0], fun=tangential_trace(k_ext, polarization, direction))
    magnetic_incident = bempp.api.GridFunction(lhs_op.domain_spaces[1], fun=neumann_trace(k_ext, polarization, direction))

    rhs = rhs_op * [electric_incident, magnetic_incident]
    b = projections_of_grid_function_list(rhs, lhs_op.dual_to_range_spaces)
    lhs_op_wf = lhs_op.weak_form()
    A = lhs_op_wf
    f = b
    if setting == 'u':
        print('solver unpreconditioned')
        x, info, res, timing = gmres(lhs_op_wf, b, return_residuals=True, restart=restart, maxiter=maxiter, tol=tolerance)
    else:
        AA = DenseTensorLinearOperator(A, A)
        ff = np.dot(np.array([f]).T, np.array([f])).ravel()
        print('solver Unpreconditioned')
        x, info, res, timing= gmres( AA ,ff, return_residuals=True,restart=restart, maxiter=maxiter, tol=tolerance)

elif setting in ['c', 'C', 'c2', 'C2']:
    transmission_operators = assemble_operators(base_grid, k_int, k_ext, far_field_points, spaces='maxwell')
    multitrace_int, multitrace_ext, identity, electric_far, magnetic_far = transmission_operators
    rescaled_int_op = rescale(multitrace_int, np.sqrt(eps_rel), np.sqrt(mu_rel))
    lhs_op = rescaled_int_op + multitrace_ext
    rhs_op = .5 * identity - rescaled_int_op
    electric_incident = bempp.api.GridFunction(lhs_op.domain_spaces[0], fun=tangential_trace(k_ext, polarization, direction),
                                              dual_space=lhs_op.dual_to_range_spaces[0])
    magnetic_incident = bempp.api.GridFunction(lhs_op.domain_spaces[1], fun=neumann_trace(k_ext, polarization, direction),
                                              dual_space=lhs_op.dual_to_range_spaces[1])
    rhs = rhs_op * [electric_incident, magnetic_incident]
    b = coefficients_of_grid_function_list(rhs)
    lhs_op_sf = lhs_op.strong_form()

    A = lhs_op_sf
    f = b
    if setting == 'c':
        print('solver calderon')
        x, info, res, timing = gmres(A,f, return_residuals=True, restart=restart, maxiter=maxiter, tol=tolerance)

    elif setting == 'C':
        AA = DenseTensorLinearOperator(A, A)
        ff = np.dot(np.array([f]).T, np.array([f])).ravel()
        print('solver Calderon')
        x, info, res, timing = gmres( AA ,ff, return_residuals=True,restart=restart, maxiter=maxiter, tol=tolerance)
    
    else:
        A = (lhs_op * lhs_op).strong_form()
        f = coefficients_of_grid_function_list(lhs_op * rhs)
        
        if setting == 'c2':
            print('solver calderon 2')
            x, info, res, timing = gmres(A,f, return_residuals=True, restart=restart, maxiter=maxiter, tol=tolerance)

        else:
            AA = DenseTensorLinearOperator(A, A)
            ff = np.dot(np.array([f]).T, np.array([f])).ravel()
            print('solver Calderon 2')
            x, info, res, timing = gmres( AA ,ff, return_residuals=True,restart=restart, maxiter=maxiter, tol=tolerance)


print(time.time() - ta, 'TEXEC')

name = (
"results/"
    + "precision"
+ str(precision)
+ "_case"
+ str(case) 
+ "_setting"
+ str(setting) 
+ "_maxiter"
+ str(maxiter))

my_dict = {
        "res": res,
        "timing": timing}

if save:
    np.save(name + ".npy", my_dict)        