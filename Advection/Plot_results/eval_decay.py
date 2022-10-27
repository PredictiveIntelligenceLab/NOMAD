import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from jax import random, jit, vmap

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

Nt = 100
Nx = 256
N= 2000

d = np.load("../Data/pure_advection_traintest.npz")
curve_fs = d["solution"][:,:,:].reshape(N,Nt*Nx,1)

curve_fs = curve_fs - np.mean(curve_fs,axis=0)

def fPCA_eig(functions, gram_cov):
    gramevals, gramevecs = np.linalg.eigh(gram_cov)
    efuncs = np.matmul(gramevecs, functions.T).T
    evals, efuncs = np.flip(gramevals, axis=-1), np.flip(efuncs, axis=-1)
    return evals, efuncs

from sklearn.metrics.pairwise import pairwise_distances
gram_mat2 = (1./(Nt*Nx))*pairwise_distances(curve_fs[:,:,0],metric=np.dot)

print(gram_mat2.shape)
evals_rho, efuncs_rho = fPCA_eig(curve_fs[:,:,0].T, gram_mat2[:,:])

fig = plt.figure(figsize=(6,5))
plt.plot(evals_rho, 'k')
plt.xlabel(r'Dimension index')
plt.ylabel(r'Eigenvalue')
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig("advection_decay.png", bbox_inches='tight', dpi=600)