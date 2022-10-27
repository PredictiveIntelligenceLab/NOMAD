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

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18,8))

@jit
def analytical_solution(key, x, t):
    mu = random.uniform(key, minval=0.05, maxval=1.0)
    return vmap(initial_condition,in_axes=(None,0,None))(x,t,mu), initial_condition(x,0,mu)

def initial_condition(x, t, mu):
    x = x-c*t
    denom = 1./jnp.sqrt(0.0002*jnp.pi)
    return denom*jnp.exp(-(1./0.0002)*(x-mu)**2)

lb_x = 0.
ub_x = 2

lb_t = 0
ub_t = 1

Nt = 1024
Nx = 1024
N = 2000

x = jnp.linspace(0,2,num=Nx)
t = jnp.linspace(0,1,num=Nt)
grid = jnp.meshgrid(x, t)
c = 1

keys = random.split(random.PRNGKey(1000),num=N)
T_exact_all, u_exact_all = vmap(analytical_solution,in_axes=(0,None,None))(keys,x,t)


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(vmin=0, vmax=10))

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

idx=33
ax.plot(x, 0*t, T_exact_all[idx,0,:], 'k')
ax.plot_wireframe(grid[0], grid[1], T_exact_all[idx,:,:], rstride=64, cstride=0, color='k', alpha=0.4)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'$s(x,t)$')
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
ax.zaxis.labelpad = 10

plt.tight_layout()
plt.savefig("advection_solution.png", bbox_inches='tight', dpi=600)