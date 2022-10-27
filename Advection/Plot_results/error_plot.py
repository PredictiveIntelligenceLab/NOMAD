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
n_hat = [2, 10,  20, 30, 40, 50, 60, 70, 80, 90, 100]
iterations = [1, 2, 3, 4, 5, 6, 7]
test_error_DON_linear = np.zeros((len(n_hat), len(iterations)))
test_error_DON_nonlinear = np.zeros((len(n_hat), len(iterations)))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18,8))
for i in range(len(n_hat)):
    n = n_hat[i]
    for j in range(len(iterations)):
        it = iterations[j]
        d = np.load("../Error_Vectors/Error_Advection_DeepONet_nhat%d_iteration%d_linear.npz"%(n,it))
        test_error_DON_linear[i,j]   = np.mean(d["test_error"])

        d = np.load("../Error_Vectors/Error_Advection_DeepONet_nhat%d_iteration%d_nonlinear.npz"%(n,it))
        test_error_DON_nonlinear[i,j]   = np.mean(d["test_error"])

lin_mu, lin_std = np.median(test_error_DON_linear, axis = 1), stats.median_abs_deviation(test_error_DON_linear, axis = 1)
nonlin_mu, nonlin_std = np.median(test_error_DON_nonlinear, axis = 1), stats.median_abs_deviation(test_error_DON_nonlinear, axis = 1)

dispersion_scale = 1.0
lin_lower = np.log10(np.clip(lin_mu - dispersion_scale*lin_std, a_min=0., a_max = np.inf) + 1e-8)
lin_upper = np.log10(lin_mu + dispersion_scale*lin_std + 1e-8)

nonlin_lower = np.log10(np.clip(nonlin_mu - dispersion_scale*nonlin_std, a_min=0., a_max = np.inf) + 1e-8)
nonlin_upper = np.log10(nonlin_mu + dispersion_scale*nonlin_std + 1e-8)

fig = plt.figure(figsize=(6,5))
plt.plot(np.array(n_hat), np.log10(lin_mu), 'k', label='Linear decoder')
plt.fill_between(np.array(n_hat), lin_lower, lin_upper, 
                         facecolor='black', alpha=0.5)

plt.plot(np.array(n_hat), np.log10(nonlin_mu), 'm', label='NOMAD')
plt.fill_between(np.array(n_hat), nonlin_lower, nonlin_upper, 
                         facecolor='magenta', alpha=0.5)
plt.legend(frameon=False)
plt.xlabel(r'Latent dimension $n$')
plt.ylabel(r'Relative $\mathcal{L}_2$ error ($\log_{10}$)')
plt.xticks(n_hat)
plt.savefig("advection_errors.png", bbox_inches='tight', dpi=600)

