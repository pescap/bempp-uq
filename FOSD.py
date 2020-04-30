import bempp.api as bem
import numpy as np
from matplotlib import pyplot as plt

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
		if x <= 0.5 / 3.:
			zx = np.sin(x * np.pi * 6)
	if i == 4:
		if x > 0.5 / 3. and x <= 1 / 3.:
			zx = -np.sin(x * np.pi * 6)
	if i ==5:
		if x > 1/3.:
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
		if y <= 0.5 / 3.:
			zy = np.sin(y * np.pi * 6)

	if j == 4:
		if y > 0.5 / 3. and y <= 1 / 3.:
			zy = -np.sin(y * np.pi * 6)
	if j == 5:
		if y > 1/3.:
			zy = np.sin(y * np.pi * 6)
	
	return zx * zy


# Def random:

def Phiz(point, n, domain_index, result):
	x, y, z = point

	res = 0j
	if z == 0.5 and (x <= 0.5) and (y <= 0.5):
		for ii in range(6):
			for jj in range(6):
				res += Random[ii,jj] * function(x, y, ii,jj)	
	result[0] = res



def perturbate(grid, t, kappa_pert=None):
    P1 = bem.function_space(grid, 'P', 1)
    grid_funz = bem.GridFunction(P1, fun = Phiz)
    elements = grid.leaf_view.elements
    vertices = grid.leaf_view.vertices
    normals = P1.global_dof_normals
    x, y, z = vertices
    
    vertices[2, :] = z + t * grid_funz.coefficients
    return bem.grid_from_element_data(vertices, elements)
    
def dirichlet_fun(x, n, domain_index, result):
    result[0] = np.exp(1j * k0 * x[0])

def neumann_fun(x, n, domain_index, result):
    result[0] = 1j * k0 * np.exp(1j * k0 * x[0]) * n[0]


k0 = 1
precision = 10
t = .075
tolerance = 1e-4


h = 2.0 * np.pi / (precision * k0)
h = 1.
#gr1 = gridL0.clone()
gr0 = bem.shapes.reentrant_cube(h=h)

elements0 = list(gr0.leaf_view.entity_iterator(0))
N0 = len(elements0)
print N0, 'N00'
for i in range(N0):
	el0 = elements0[i]
	gr0.mark(el0)
gr1 = gr0.refine()

elements0 = list(gr1.leaf_view.entity_iterator(0))
N0 = len(elements0)
print N0, 'N01'
for i in range(N0):
	el0 = elements0[i]
	gr1.mark(el0)
grid = gr1.refine()


space = bem.function_space(grid, 'DP', 0)	
N = int(space.global_dof_count)
theta = np.linspace(0, 2 * np.pi, 400)
points = np.array([np.cos(theta), np.sin(theta), np.zeros(len(theta))])

######
## FOSB
######

corr = np.zeros((N, N), dtype= np.complex128)

for ii in range(5):
	for jj in range(5):
		def fun(point, n, domain_index, result):
			x, y, z = point

			res = 0j
			if z == 0.5 and (x <= 0.5) and (y <= 0.5):
				res += function(x, y, ii,jj)
			result[0] = res


		Cx =  np.array([bem.GridFunction(space, fun= fun).coefficients])
		corr += 1./3 * np.kron(Cx, Cx.T)

refvar = corr.diagonal()


#######

from login import  assemble_singular_part
from ilu import InverseSparseDiscreteBoundaryOperator

V = bem.operators.boundary.helmholtz.single_layer(space, space, space, k0)
V_sing_wf = assemble_singular_part(V, distance = 4)
A = InverseSparseDiscreteBoundaryOperator(V_sing_wf, mu = 0.01)

df =  bem.GridFunction(space , fun = dirichlet_fun)

phi, info, res = bem.linalg.gmres(V, df, tol=tolerance, return_residuals=True, use_strong_form=False, maxiter=1000, restart=1000)

Vm = bem.as_matrix(V.weak_form())

rhsp = np.array([-phi.projections()])
rhs2 = np.kron(rhsp, np.conj(rhsp.T)) * corr

Vm1 = np.linalg.pinv(Vm)
sigma = np.dot(np.dot(Vm1, rhs2), np.conj(Vm1).T)


from scipy.sparse.linalg import LinearOperator
class DenseTensorLinearOperator(LinearOperator):
    
    def __init__(self, A0, A1):
        self.A0 = A0
        self.A1 = A1
        self.n0 = self.A0.shape[0]
        self.n1 = self.A1.shape[0]
        self.shape = self.n0*self.n1, self.n0*self.n1
        self.dtype = self.A0.dtype

    def _matvec(self, x):
        n0 = self.A0.shape[0]
        n1 = self.A1.shape[0]
        xbis0 = x.reshape([n0, n1])
        ytemp = self.A0 * xbis0
        ztemp = (self.A1 * ytemp.T).T.reshape(n0*n1)
        return ztemp


#### Get Sigma bis....

V_sing_wf = assemble_singular_part(V, distance = 4)
C = InverseSparseDiscreteBoundaryOperator(V_sing_wf, mu = 0.01)



V_wf = V.weak_form()
A0 = C * V_wf
A1 = (C * V_wf)._impl.conjugate()
A1 = LinearOperator( A1.shape, matvec=A1.matvec)

AA = DenseTensorLinearOperator(A0, A1)
from login import gmres
X, info, res = gmres( AA ,np.ravel(rhs2), tol=tolerance, return_residuals=True,  maxiter=1000, restart=1000)
sigma2 = X.reshape((N, N))


slp_far_field = bem.operators.far_field.helmholtz.single_layer(space, points, k0)
SLP = bem.as_matrix(slp_far_field.discrete_operator)

U = np.array([-np.dot(SLP, phi.coefficients)])

UU = np.dot(np.dot(SLP, sigma), np.conj(SLP).T)

UU2 = np.dot(np.dot(SLP, sigma2), np.conj(SLP).T)


variance = t **2 * np.array([UU.diagonal() ])
variance2 = t **2 * np.array([UU2.diagonal() ])



#output = k0, precision, N, M, Umean, Uvar, U, variance


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

########
# ax1, ax3: MC
########
bistatic_rcs= 10 * np.log10(4 * np.pi * np.sum( ( np.sqrt(variance2)) **2, axis=0))
ax3.plot(theta, bistatic_rcs, '--' ,label='+Var', color='b')


#bistatic_rcs= 10 * np.log10(4 * np.pi * np.sum( ( np.sqrt(variance)) **2, axis=0))
bistatic_rcs= 10 * np.log10(4 * np.pi * np.sum( ( np.sqrt(variance)) **2, axis=0))
ax4.plot(theta, bistatic_rcs, '--' ,label='+Var', color='b')
#ax2.legend()
#ax2.set_title("Bistatic RCS [dB]")
#ax2.set_xlabel('Angle (Degrees)')

for ax in fig.get_axes():
    ax.label_outer()

plt.show(block=False)
