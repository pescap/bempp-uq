#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bempp.api
import numpy as np
from bempp.api.assembly.blocked_operator import (
    coefficients_of_grid_function_list,
    grid_function_list_from_coefficients,
    projections_of_grid_function_list,
)

from bemppUQ.assembly.operators import DenseTensorLinearOperator
from bemppUQ.functions import neumann_trace, tangential_trace
from bemppUQ.operators.maxwell import assemble_operators
from bemppUQ.shapes import kite
from bemppUQ.utils.login import gmres, rescale

restart = 100
maxiter = 100
tolerance = 1e-4

number_of_angles = 400
angles = np.pi * np.linspace(0, 2, number_of_angles)
far_field_points = np.array(
    [np.cos(angles), np.sin(angles), np.zeros(number_of_angles)]
)
case = "LF"

if case == "LF":
    polarization = np.array([1.0 + 1j, 2.0, -1.0 - 1.0 / 3.0 * 1j])
    direction = np.array(
        [1.0 / np.sqrt(14), 2.0 / np.sqrt(14), 3.0 / np.sqrt(14)], dtype="float64"
    )

    eps_rel = 1.9
    mu_rel = 1.0

    k_ext = 1.047197551196598
    k_int = k_ext * np.sqrt(eps_rel * mu_rel)


print("The exterior wavenumber is: {0}".format(k_ext))
print("The interior wavenumber is: {0}".format(k_int))

precision = 30
h = 2.0 * np.pi / (precision * k_int)
base_grid = kite(h=h)

print(base_grid.leaf_view.entity_count(1) * 2, "Ndof")
# Primal formaulation
transmission_operators = assemble_operators(
    base_grid, k_int, k_ext, far_field_points, spaces="maxwell_primal", osrc=True
)

(
    multitrace_int,
    multitrace_ext,
    osrc_int,
    osrc_ext,
    identity,
    electric_far,
    magnetic_far,
) = transmission_operators

rescaled_osrc_int_op = rescale(osrc_int, np.sqrt(eps_rel), np.sqrt(mu_rel))
rescaled_int_op = rescale(multitrace_int, np.sqrt(eps_rel), np.sqrt(mu_rel))

lhs_osrc = rescaled_osrc_int_op + osrc_ext
lhs_op = rescaled_int_op + multitrace_ext
rhs_op = 0.5 * identity - rescaled_int_op
electric_incident = bempp.api.GridFunction(
    lhs_op.domain_spaces[0], fun=tangential_trace(k_ext, polarization, direction)
)
magnetic_incident = bempp.api.GridFunction(
    lhs_op.domain_spaces[1], fun=neumann_trace(k_ext, polarization, direction)
)

rhs = rhs_op * [electric_incident, magnetic_incident]
b = projections_of_grid_function_list(rhs, lhs_op.dual_to_range_spaces)
lhs_op_wf = lhs_op.weak_form()
P = lhs_osrc.weak_form()

print("solver osrc")
A = P * lhs_op_wf
f = P * b
x_osrc, info, res_osrc, times_osrc = gmres(
    A, f, return_residuals=True, restart=restart, maxiter=maxiter, tol=tolerance
)
solution_osrc = grid_function_list_from_coefficients(x_osrc, lhs_op.domain_spaces)
far_field_osrc = (
    4.0 * np.pi * (-electric_far * solution_osrc[1] - magnetic_far * solution_osrc[0])
)


AA = DenseTensorLinearOperator(A, A)
ff = np.dot(np.array([f]).T, np.array([f])).ravel()
print("solver Osrc")
X_osrc, info, Res_osrc, Times_osrc = gmres(
    AA, ff, return_residuals=True, restart=restart, maxiter=maxiter, tol=tolerance
)


print("solver unpreconditioned")
A = lhs_op_wf
f = b
x_unprec, info, res_unprec, times_unprec = gmres(
    lhs_op_wf, b, return_residuals=True, restart=restart, maxiter=maxiter, tol=tolerance
)
solution_unprec = grid_function_list_from_coefficients(x_unprec, lhs_op.domain_spaces)
far_field_unprec = (
    4.0
    * np.pi
    * (-electric_far * solution_unprec[1] - magnetic_far * solution_unprec[0])
)

AA = DenseTensorLinearOperator(A, A)
ff = np.dot(np.array([f]).T, np.array([f])).ravel()

print("solver Unpreconditioned")
X_unprec, info, Res_unprec, Times_unprec = gmres(
    AA, ff, return_residuals=True, restart=restart, maxiter=maxiter, tol=tolerance
)


print("solver dual")
transmission_operators = assemble_operators(
    base_grid, k_int, k_ext, far_field_points, spaces="maxwell"
)
(
    multitrace_int,
    multitrace_ext,
    identity,
    electric_far,
    magnetic_far,
) = transmission_operators
rescaled_int_op = rescale(multitrace_int, np.sqrt(eps_rel), np.sqrt(mu_rel))
lhs_op = rescaled_int_op + multitrace_ext
rhs_op = 0.5 * identity - rescaled_int_op
electric_incident = bempp.api.GridFunction(
    lhs_op.domain_spaces[0],
    fun=tangential_trace(k_ext, polarization, direction),
    dual_space=lhs_op.dual_to_range_spaces[0],
)
magnetic_incident = bempp.api.GridFunction(
    lhs_op.domain_spaces[1],
    fun=neumann_trace(k_ext, polarization, direction),
    dual_space=lhs_op.dual_to_range_spaces[1],
)
rhs = rhs_op * [electric_incident, magnetic_incident]
b = coefficients_of_grid_function_list(rhs)
lhs_op_sf = lhs_op.strong_form()

A = lhs_op_sf
f = b

print("solver calderon")
x_cald, info, res_cald, times_cald = gmres(
    A, f, return_residuals=True, restart=restart, maxiter=maxiter, tol=tolerance
)
solution_cald = grid_function_list_from_coefficients(x_cald, lhs_op.domain_spaces)
far_field_cald = (
    4.0 * np.pi * (-electric_far * solution_cald[1] - magnetic_far * solution_cald[0])
)

AA = DenseTensorLinearOperator(A, A)
ff = np.dot(np.array([f]).T, np.array([f])).ravel()

print("solver Calderon")
X_cald, info, Res_cald, Times_cald = gmres(
    AA, ff, return_residuals=True, restart=restart, maxiter=maxiter, tol=tolerance
)


A = (lhs_op * lhs_op).strong_form()
f = coefficients_of_grid_function_list(lhs_op * rhs)

print("solver calderon 2")
x_cald2, info, res_cald2, times_cald2 = gmres(
    A, f, return_residuals=True, restart=restart, maxiter=maxiter, tol=tolerance
)
solution_cald = grid_function_list_from_coefficients(x_cald, lhs_op.domain_spaces)
far_field_cald = (
    4.0 * np.pi * (-electric_far * solution_cald[1] - magnetic_far * solution_cald[0])
)

AA = DenseTensorLinearOperator(A, A)
ff = np.dot(np.array([f]).T, np.array([f])).ravel()

print("solver Calderon 2")
X_cald, info, Res_cald2, Times_cald2 = gmres(
    AA, ff, return_residuals=True, restart=restart, maxiter=maxiter, tol=tolerance
)

name = (
    "results/"
    + "precision"
    + str(precision)
    + "_case"
    + str(case)
    + "_maxiter"
    + str(maxiter)
    + "30"
)

my_dict = {
    "res_unprec": res_unprec,
    "res_osrc": res_osrc,
    "res_cald": res_cald,
    "res_cald2": res_cald2,
    "Res_unprec": res_unprec,
    "Res_osrc": res_osrc,
    "Res_cald": res_cald,
    "Res_cald2": res_cald2,
    "times_unprec": times_unprec,
    "times_osrc": times_osrc,
    "times_cald": times_cald,
    "times_cald2": times_cald2,
    "Times_unprec": Times_unprec,
    "Times_osrc": Times_osrc,
    "Times_cald": Times_cald,
    "Times_cald2": Times_cald2,
}


np.save(name + ".npy", my_dict)
