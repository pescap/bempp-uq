import bempp.api as bem
import numpy as np
import scipy.sparse.linalg
from bempp.api.assembly import GridFunction
from bempp.api.assembly import BoundaryOperator

def get_h(grid):
    # routine that gives the exact value of h defined as the maximum length of the triangles
    # of a given mesh
    elements1 = list(grid.leaf_view.entity_iterator(1))
    vol = []
    for el1 in elements1:
        vol.append(el1.geometry.volume)
    vol = np.array(vol)
    return [vol.min(), vol.max(), vol.mean()]


def relative_error(self, fun, element=None):
    """Compute the relative L^2 error compared to a given analytic function."""
    from bempp.api.integration import gauss_triangle_points_and_weights
    import numpy as np
    p_normal = np.array([[0]])
    global_diff = 0
    fun_l2_norm = 0
    accuracy_order = self.parameters.quadrature.far.single_order
    points, weights = gauss_triangle_points_and_weights(accuracy_order)
    npoints = points.shape[1]
    element_list = [element] if element is not None else list(self.grid.leaf_view.entity_iterator(0))
    for element in element_list:
        integration_elements = element.geometry.integration_elements(points)
        normal = element.geometry.normals(p_normal).reshape(3)
        global_dofs = element.geometry.local2global(points)
        fun_vals = np.zeros((self.component_count, npoints), dtype=self.dtype)
        for j in range(npoints):
            fun_vals[:, j] = fun(global_dofs[:, j], normal)
        diff = np.sum(np.abs(self.evaluate(element, points) - fun_vals)**2, axis=0)
        global_diff += np.sum(diff * integration_elements * weights)
        abs_fun_squared = np.sum(np.abs(fun_vals)**2, axis=0)
        fun_l2_norm += np.sum(abs_fun_squared * integration_elements * weights)
    return np.sqrt(global_diff/fun_l2_norm)


class _it_counter(object):

    def __init__(self, store_residuals):
        self._count = 0
        self._store_residuals = store_residuals
        self._residuals = []

    def __call__(self, x):
        self._count += 1
        if self._store_residuals:
            self._residuals.append(np.linalg.norm(x))
            #print('iteration -', self._count,"|| residual -", np.linalg.norm(x))


    @property
    def count(self):
        return self._count

    @property
    def residuals(self):
        return self._residuals

def gmres(A, b, tol=1E-5, restart=None, maxiter=None, use_strong_form=False, return_residuals=False):
    """Interface to the scipy.sparse.linalg.gmres function.

    This function behaves like the scipy.sparse.linalg.gmres function. But
    instead of a linear operator and a vector b it can take a boundary operator
    and a grid function. In this case, the result is returned as a grid function in the
    correct space.
    
    """
    import bempp.api
    import time

    if not isinstance(A, BoundaryOperator) and use_strong_form==True:
        raise ValueError("Strong form only with BoundaryOperator")
    if isinstance(A, BoundaryOperator) and not isinstance(b, GridFunction):
        raise ValueError("Instance Error")
    

    # Assemble weak form before the logging messages

    if isinstance(A, BoundaryOperator) and isinstance(b, GridFunction):
        if use_strong_form:
            if not A.range.is_compatible(b.space):
                raise ValueError(
                    "The range of A and the space of A must have the same number of unknowns if the strong form is used.")
            A_op = A.strong_form()
            b_vec = b.coefficients
        else:
            A_op = A.weak_form()
            b_vec = b.projections(A.dual_to_range)
    else:
        A_op = A
        b_vec = b

    callback = _it_counter(return_residuals)

    
    start_time = time.time()
    x, info = scipy.sparse.linalg.gmres(A_op, b_vec,
                                        tol=tol, restart=restart, maxiter=maxiter, callback=callback)
    end_time = time.time()
    

    if isinstance(A, BoundaryOperator) and isinstance(b, GridFunction):
        res_fun = GridFunction(A.domain, coefficients=x.ravel())

    else:
        res_fun = x
    if return_residuals:
        return res_fun, info, callback.residuals
    else:
        return res_fun, info
