import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

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

# load the dataset
n_hat = [1, 2, 5, 10, 30, 50, 70, 100]
iterations = [0,1, 2, 3, 4, 5, 6, 7,8,9]
par = 0
test_error_DON_linear = np.zeros((len(n_hat), len(iterations)))
test_error_DON_nonlinear = np.zeros((len(n_hat), len(iterations)))

for i in range(len(n_hat)):
    n = n_hat[i]
    for j in range(len(iterations)):
        it = iterations[j]
        d = np.load("../Error_vectors/Error_SW_DeepONet_nhat%d_iteration%d_linear.npz"%(n,it))
        test_error_DON_linear[i,j]   = np.mean(d["test_error"],axis=1)[...,par]

        d = np.load("../Error_vectors/Error_SW_DeepONet_nhat%d_iteration%d_nonlinear.npz"%(n,it))
        test_error_DON_nonlinear[i,j]   = np.mean(d["test_error"],axis=1)[...,par]

lin_mu, lin_std = np.median(test_error_DON_linear, axis = 1), stats.median_abs_deviation(test_error_DON_linear, axis = 1)
nonlin_mu, nonlin_std = np.median(test_error_DON_nonlinear, axis = 1), stats.median_abs_deviation(test_error_DON_nonlinear, axis = 1)

dispersion_scale = 1.0
lin_lower = np.log10(np.clip(lin_mu - dispersion_scale*lin_std, a_min=0., a_max = np.inf) + 1e-8)
lin_upper = np.log10(lin_mu + dispersion_scale*lin_std + 1e-8)

nonlin_lower = np.log10(np.clip(nonlin_mu - dispersion_scale*nonlin_std, a_min=0., a_max = np.inf) + 1e-8)
nonlin_upper = np.log10(nonlin_mu + dispersion_scale*nonlin_std + 1e-8)

fig = plt.figure(figsize=(8,7))
ax1 = fig.add_subplot(3,1,1)
ax1.plot(np.array(n_hat), np.log10(lin_mu), 'k', label='Linear Decoder')
ax1.fill_between(np.array(n_hat), lin_lower, lin_upper, 
                         facecolor='black', alpha=0.5), 100

ax1.plot(np.array(n_hat), np.log10(nonlin_mu), 'm', label='NOMAD')
ax1.fill_between(np.array(n_hat), nonlin_lower, nonlin_upper, 
                         facecolor='magenta', alpha=0.5)
ax1.legend(frameon=False)
ax1.set_xticks([1,5,10,30, 50, 70, 100])
axR1 = fig.add_subplot(3,1,1, sharex=ax1, frameon=False)
axR1.yaxis.tick_right()
axR1.yaxis.set_label_position("right")
axR1.axes.yaxis.set_ticklabels([])
axR1.set_ylabel(r'$\rho$')

test_error_DON_linear = np.zeros((len(n_hat), len(iterations)))
test_error_DON_nonlinear = np.zeros((len(n_hat), len(iterations)))
par = 1
for i in range(len(n_hat)):
    n = n_hat[i]
    for j in range(len(iterations)):
        it = iterations[j]
        d = np.load("../Error_vectors/Error_SW_DeepONet_nhat%d_iteration%d_linear.npz"%(n,it))
        test_error_DON_linear[i,j]   = np.mean(d["test_error"],axis=1)[...,par]

        d = np.load("../Error_vectors/Error_SW_DeepONet_nhat%d_iteration%d_nonlinear.npz"%(n,it))
        test_error_DON_nonlinear[i,j]   = np.mean(d["test_error"],axis=1)[...,par]

lin_mu, lin_std = np.median(test_error_DON_linear, axis = 1), stats.median_abs_deviation(test_error_DON_linear, axis = 1)
nonlin_mu, nonlin_std = np.median(test_error_DON_nonlinear, axis = 1), stats.median_abs_deviation(test_error_DON_nonlinear, axis = 1)

dispersion_scale = 1.0
lin_lower = np.log10(np.clip(lin_mu - dispersion_scale*lin_std, a_min=0., a_max = np.inf) + 1e-8)
lin_upper = np.log10(lin_mu + dispersion_scale*lin_std + 1e-8)

nonlin_lower = np.log10(np.clip(nonlin_mu - dispersion_scale*nonlin_std, a_min=0., a_max = np.inf) + 1e-8)
nonlin_upper = np.log10(nonlin_mu + dispersion_scale*nonlin_std + 1e-8)

ax2 = fig.add_subplot(3,1,2)
ax2.plot(np.array(n_hat), np.log10(lin_mu), 'k', label='Linear Decoder')
ax2.fill_between(np.array(n_hat), lin_lower, lin_upper, 
                         facecolor='black', alpha=0.5), 100

ax2.plot(np.array(n_hat), np.log10(nonlin_mu), 'm', label='NOMAD')
ax2.fill_between(np.array(n_hat), nonlin_lower, nonlin_upper, 
                         facecolor='magenta', alpha=0.5)
ax2.legend(frameon=False)
ax2.set_ylabel(r'Relative $\mathcal{L}_2$ error ($\log_{10}$)')
ax2.set_xticks([1,5,10,30, 50, 70, 100])
axR2 = fig.add_subplot(3,1,2, sharex=ax2, frameon=False)
axR2.yaxis.tick_right()
axR2.yaxis.set_label_position("right")
axR2.axes.yaxis.set_ticklabels([])
axR2.set_ylabel(r'$v_1$')

test_error_DON_linear = np.zeros((len(n_hat), len(iterations)))
test_error_DON_nonlinear = np.zeros((len(n_hat), len(iterations)))
par = 2

for i in range(len(n_hat)):
    n = n_hat[i]
    for j in range(len(iterations)):
        it = iterations[j]
        d = np.load("../Error_vectors/Error_SW_DeepONet_nhat%d_iteration%d_linear.npz"%(n,it))
        test_error_DON_linear[i,j]   = np.mean(d["test_error"],axis=1)[...,par]

        d = np.load("../Error_vectors/Error_SW_DeepONet_nhat%d_iteration%d_nonlinear.npz"%(n,it))
        test_error_DON_nonlinear[i,j]   = np.mean(d["test_error"],axis=1)[...,par]

lin_mu, lin_std = np.median(test_error_DON_linear, axis = 1), stats.median_abs_deviation(test_error_DON_linear, axis = 1)
nonlin_mu, nonlin_std = np.median(test_error_DON_nonlinear, axis = 1), stats.median_abs_deviation(test_error_DON_nonlinear, axis = 1)

dispersion_scale = 1.0
lin_lower = np.log10(np.clip(lin_mu - dispersion_scale*lin_std, a_min=0., a_max = np.inf) + 1e-8)
lin_upper = np.log10(lin_mu + dispersion_scale*lin_std + 1e-8)

nonlin_lower = np.log10(np.clip(nonlin_mu - dispersion_scale*nonlin_std, a_min=0., a_max = np.inf) + 1e-8)
nonlin_upper = np.log10(nonlin_mu + dispersion_scale*nonlin_std + 1e-8)

ax3 = fig.add_subplot(3,1,3)
ax3.plot(np.array(n_hat), np.log10(lin_mu), 'k', label='Linear Decoder')
ax3.fill_between(np.array(n_hat), lin_lower, lin_upper, 
                         facecolor='black', alpha=0.5), 100

ax3.plot(np.array(n_hat), np.log10(nonlin_mu), 'm', label='NOMAD')
ax3.fill_between(np.array(n_hat), nonlin_lower, nonlin_upper, 
                         facecolor='magenta', alpha=0.5)
ax3.legend(frameon=False)
ax3.set_xlabel(r'Latent dimension $n$')
ax3.set_xticks([1,5,10,30, 50, 70, 100])
axR3 = fig.add_subplot(3,1,3, sharex=ax3, frameon=False)
axR3.yaxis.tick_right()
axR3.yaxis.set_label_position("right")
axR3.axes.yaxis.set_ticklabels([])
axR3.set_ylabel(r'$v_2$')

plt.savefig("SW_errors.png", bbox_inches='tight', dpi=600)

