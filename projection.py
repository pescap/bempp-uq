import bempp.api as bem
import numpy as np
import scipy.spatial
from scipy.sparse import lil_matrix

def get_h(grid):
    # routine that gives the exact value of h defined as the maximum length of the triangles
    # of a given mesh
    elements1 = list(grid.leaf_view.entity_iterator(1))
    vol = []
    for el1 in elements1:
        vol.append(el1.geometry.volume)
    vol = np.array(vol)
    return [vol.min(), vol.max(), vol.mean()]

   
def discrete_coefficients_projection(sp0, sp1):
	# Discrete projection from sp0 to sp1, with sp1 obtained with the refine function of bempp
	# For P1 space

	gr0 = sp0.grid
	gr1 = sp1.grid
	n0 = int(sp0.global_dof_count)
	n1 = int(sp1.global_dof_count)

	mid = []
	info = []
	triangles0 = list(gr0.leaf_view.entity_iterator(0))

	for el0 in triangles0:
		dof = sp0.get_global_dofs(el0)
		a,b,c = el0.geometry.corners.T

		m1 = (a + b) / 2.
		m2 = (b + c) / 2.
		m3 = (c + a) / 2.

		mid.append(m1)
		mid.append(m2)
		mid.append(m3)

		info.append([dof[0], dof[1]])
		info.append([dof[1], dof[2]])
		info.append([dof[2], dof[0]])


	#h = gr1.leaf_view.minimum_element_diameter
	h = get_h(gr1)[0]
	tolerance = h / 10.

	v1 = gr1.leaf_view.vertices.T
	vertices1 =  [v1[i] for i in range(len(v1))]

	Tree = scipy.spatial.KDTree(v1)
	distance, index = Tree.query(mid)

	triangleMap = np.repeat(None,len(index))
	triangleMap[distance < tolerance] = index[distance< tolerance]

	val = []
	for i in range(triangleMap.shape[0]):
		if triangleMap[i] is not None:
			val.append((triangleMap[i],info[i][0]))
			val.append((triangleMap[i],info[i][1]))
			


	P =  lil_matrix((n1,n0))
	P.setdiag(1)
	for vali in val:
		P[vali] = 1/2.
	return P.tocsr()



def dirichlet_fun(x, n, domain_index, result):
    result[0] = np.exp(1j * k0 * x[0])

def neumann_fun(x, n, domain_index, result):
    result[0] = 1j * k0 * np.exp(1j * k0 * x[0]) * n[0] 


def refine(grid, tosphere=0):
	gr1 = grid.clone()
	el0 = list(grid.leaf_view.entity_iterator(0))
	[gr1.mark(el) for el in el0]
	#grid = grid.refine()
	gr1.refine()
	vertices = gr1.leaf_view.vertices
	if tosphere == 1:
		coeff = np.sqrt((vertices **2).sum(axis=0))
		vertices /= coeff
	gr1 = bem.grid_from_element_data(vertices,gr1.leaf_view.elements)
	return gr1


if __name__ == "__main__":

	from kite import kite

	k0 = 1
	precision = 20
	tolerance = 1E-4
	h = 2.0 * np.pi / (precision * k0)

	h = 0.2
	#gr0 = bem.shapes.cube(h=h)
	gr0 = kite(h=h)
	gr1 = refine(gr0)
	sp0 = bem.function_space(gr0, 'P', 1)
	sp1 = bem.function_space(gr1, 'P', 1)


	u0 = bem.GridFunction(sp0, fun = dirichlet_fun)
	#u0.plot(mode = 'node', transformation='real')
	c0 = u0.coefficients

	P = discrete_coefficients_projection(sp0, sp1)
	c1 = P.dot(c0)
	print c0.shape, c1.shape

	u1 = bem.GridFunction(sp1, coefficients=c1)
	#u1.plot(mode = 'node', transformation='real')

	print u0.l2_norm()
	print u1.l2_norm()