import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
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

def exact_eigenpairs(x, n, alpha=2.0, tau=0.1):
  idx = jnp.arange(n)+1
  evals = jnp.power((2.0 * jnp.pi * idx)**2 + tau**2, -alpha)
  efuns = jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * idx * x)
  return evals, efuns

@partial(vmap, in_axes=(1, None))
@partial(vmap, in_axes=(None, 1))
def gram(f, g):
  inner_product = lambda phi_i,phi_j: jnp.einsum('ij,i,j', jnp.diag(w), phi_i, phi_j)
  return inner_product(f, g)

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
  x = 0.5*(ub - lb)*(x + 1.0) + lb
  x = np.array(x[:,None])
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

ax2 = fig.add_subplot(1, 1, 1)
ax2.loglog(evals[:50], 'b-', alpha=1.0, label='Eigenvalue Decay')
ax2.set_xlabel(r'Index')
ax2.set_ylabel(r'Eigenvalue')
ax2.legend(loc='upper center', bbox_to_anchor=(0.60, 1),
            fancybox=True, shadow=False, ncol=1)
ax2.autoscale(tight=True)

plt.savefig("eval_decay.jpg", bbox_inches='tight', pad_inches=0,dpi=300)