import bempp.api
import numpy as np


def surface_gradient(
    domain, range_, dual_to_range, label="SURFACE_GRADIENT", parameters=None
):
    from bempp.api.operators.boundary.sparse import operator_from_functors
    from bempp.api.assembly.functors import simple_test_trial_integrand_functor
    from bempp.api.assembly.functors import surface_divergence_functor
    from bempp.api.assembly.functors import scalar_function_value_functor

    return -operator_from_functors(
        domain,
        range_,
        dual_to_range,
        surface_divergence_functor(),
        scalar_function_value_functor(),
        simple_test_trial_integrand_functor(),
        label=label,
        parameters=parameters,
    )


def surface_divergence(
    domain, range_, dual_to_range, label="SURFACE_DIVERGENCE", parameters=None
):
    from bempp.api.operators.boundary.sparse import operator_from_functors
    from bempp.api.assembly.functors import hdiv_function_value_functor
    from bempp.api.assembly.functors import simple_test_trial_integrand_functor
    from bempp.api.assembly.functors import maxwell_test_trial_integrand_functor
    from bempp.api.assembly.functors import surface_gradient_functor

    return -operator_from_functors(
        domain,
        range_,
        dual_to_range,
        surface_gradient_functor(),
        hdiv_function_value_functor(),
        simple_test_trial_integrand_functor(),
        label=label,
        parameters=parameters,
    )


def trace_transformation(
    domain, range_, dual_to_range, label="TRACE_TRANSFORMATION", parameters=None
):
    return bempp.api.operators.boundary.sparse._maxwell_identity(
        domain, range_, dual_to_range, label=label, parameters=parameters
    )


def function_product(f, g, trial_space, test_space):
    # Compute the product of two functions
    from bempp.api.integration import gauss_triangle_points_and_weights
    from bempp.api.utils import combined_type

    accuracy_order = bempp.api.global_parameters.quadrature.far.single_order
    points, weights = gauss_triangle_points_and_weights(accuracy_order)
    if f.space.grid != g.space.grid:
        raise ValueError("f and g must be defined on the same grid")
    element_list = list(test_space.grid.leaf_view.entity_iterator(0))
    dtype = combined_type(f.dtype, g.dtype)
    result = np.zeros(test_space.global_dof_count, dtype=dtype)

    for element in element_list:
        dofs, multipliers = test_space.get_global_dofs(element, dof_weights=True)
        n_local_basis_funs = len(multipliers)
        for index in range(n_local_basis_funs):
            coeffs = np.zeros(n_local_basis_funs)
            coeffs[index] = 1
            integration_elements = element.geometry.integration_elements(points)
            basis_values = test_space.evaluate_local_basis(element, points, coeffs)
            f_prod_g = f.evaluate(element, points) * g.evaluate(element, points)
            prod_times_basis = np.sum(f_prod_g * basis_values, axis=0)
            local_res = np.sum(prod_times_basis * weights * integration_elements)
            result[dofs[index]] += multipliers[index] * local_res
    return bempp.api.GridFunction(
        trial_space, dual_space=test_space, projections=result
    )
