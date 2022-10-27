import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from functools import partial

import numpy as np
from functools import partial 

from jax import vmap
from numpy.polynomial.legendre import leggauss

plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({
                      "text.usetex": True,
                      "font.family": "serif",
                      'text.latex.preamble': r'\usepackage{amsmath}',
                      'font.size': 20,
                      'lines.linewidth': 3,
                      'axes.labelsize': 22,  # fontsize for x and y labels (was 10)
                      'axes.titlesize': 24,
                      'xtick.labelsize': 20,
                      'ytick.labelsize': 20,
                      'legend.fontsize': 20,
                      'axes.linewidth': 2})


fig = plt.figure(figsize=(15,4))

def f(x, t):
  return jnp.sin(2.0*jnp.pi*t*x)

# Define pairwise inner product function 
# Takes two lists of functions and returns gram matrix of inner products
@partial(vmap, in_axes=(1, None))
@partial(vmap, in_axes=(None, 1))
def gram(f, g):
  inner_product = lambda phi_i,phi_j: jnp.einsum('ij,i,j', jnp.diag(w), phi_i, phi_j)
  return inner_product(f, g)

# given a list of functions and gram matrix of pairwise inner products, gets
# principal component functions and associated eigenvalues
def fPCA_eig(functions, gram_cov):
  gramevals, gramevecs = jnp.linalg.eigh(gram_cov)
  efuncs = jnp.matmul(gramevecs, functions.T).T
  evals, efuncs = jnp.flip(gramevals, axis=-1), jnp.flip(efuncs, axis=-1)
  return evals, efuncs

# returns quadrature nodes and weights
def legendre_quadrature_1d(n_quad, bounds=(-1.0,1.0)):
  lb, ub = bounds
  # GLL nodes and weights in [-1,1]        
  x, w = leggauss(n_quad)
  # Rescale nodes to [lb,ub]
  x = 0.5*(ub - lb)*(x + 1.0) + lb
  
  x = np.array(x[:,None])
  # Determinant of Jacobian of mapping [lb,ub]-->[-1,1]
  jac_det = 0.5*(ub-lb)
  w = np.array(w*jac_det)
  return x, w


n_quad = 500
bounds = (0, 1)
x, w = legendre_quadrature_1d(n_quad, bounds)
inner_product = lambda phi_i,phi_j: jnp.einsum('ij,i,j', jnp.diag(w), phi_i, phi_j)

t0 = 0
t1 = 10
num_funcs = 1000
ts = np.linspace(t0, t1, num_funcs)
curve_fs = vmap(lambda t: f(x, t))(ts)[...,0]

mean = np.mean(curve_fs, axis=0)
mean_curves = curve_fs - mean

curve_fs = mean_curves.T
gram_mat = gram(curve_fs, curve_fs)
evals, efuncs = fPCA_eig(curve_fs, gram_mat)

projs = vmap(lambda idx: vmap(lambda ft: inner_product(ft, efuncs[:,idx]))(curve_fs.T))(np.arange(num_funcs))
print(projs.shape)

first_proj = projs[0]
second_proj = projs[1]
third_proj = projs[2]

ax1 = fig.add_subplot(1, 3, 1,  projection='3d')
ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
ax1.set_zlabel(r'z', rotation=90)
for i in range(num_funcs-1):
    ax1.plot(first_proj[i:i+2], second_proj[i:i+2], third_proj[i:i+2], color=plt.cm.cool(i/num_funcs))
ax1.set_xlabel(r'x')
ax1.set_ylabel(r'y')
ax1.set_title(r'Projection onto principal components')

plt.savefig("spiral.jpg", bbox_inches='tight', pad_inches=0,dpi=300)