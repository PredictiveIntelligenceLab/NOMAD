import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({
                      "text.usetex": True,
                      "font.family": "serif",
                      'text.latex.preamble': r'\usepackage{amsmath}',
                      'font.size': 20,
                      'lines.linewidth': 3,
                      'axes.labelsize': 22,  
                      'axes.titlesize': 24,
                      'xtick.labelsize': 20,
                      'ytick.labelsize': 20,
                      'legend.fontsize': 20,
                      'axes.linewidth': 2})


n_hat = [1, 10,  20, 30, 40, 50, 60, 70, 80, 90, 100]
iterations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
test_error_DON_linear = np.zeros((len(n_hat), len(iterations)))
test_error_DON_nonlinear = np.zeros((len(n_hat), len(iterations)))

fig = plt.figure(figsize=(15,4))

ax3 = fig.add_subplot(1, 1, 1)
for i in range(len(n_hat)):
    n = n_hat[i]
    for j in range(len(iterations)):
        it = iterations[j]
        d = np.load("../Error_Vectors/Error_Antiderivative_DeepONet_nhat%d_iteration%d_linear.npz"%(n,it))
        test_error_DON_linear[i,j]   = np.mean(d["test_error"])

        d = np.load("../Error_Vectors/Error_Antiderivative_DeepONet_nhat%d_iteration%d_nonlinear.npz"%(n,it))
        test_error_DON_nonlinear[i,j]   = np.mean(d["test_error"])

DON_linear    = np.tile(np.array(["Linear Decoder"])[None,:],(1000,1))
DON_nonlinear = np.tile(np.array(["Non-linear Decoder"])[None,:],(1000,1))

position = np.tile(np.array(['1','10','20','30','40','50', '60', '70', '80', '90', '100'])[None,:],(1000,1))

DON_all_linear    = list(zip(test_error_DON_linear.T.flatten(), DON_linear.flatten(), position.flatten()))
DON_all_nonlinear = list(zip(test_error_DON_nonlinear.T.flatten(), DON_nonlinear.flatten(), position.flatten()))

all_data = DON_all_linear + DON_all_nonlinear

df_OKA = pd.DataFrame(all_data, columns = ["Relative $\mathcal{L}_2$ error","Method", "Latent dimension size"])
flierprops = dict(markerfacecolor='0.75', markersize=0.5,
              linestyle='none')
ax3 = sns.lineplot(x="Latent dimension size", y="Relative $\mathcal{L}_2$ error", hue="Method", data=df_OKA, palette="Set1")
ax3.legend(loc='upper center', bbox_to_anchor=(0.6, 1),
            fancybox=True, shadow=False, ncol=1)
fig.tight_layout()

plt.savefig("lineplots.jpg", bbox_inches='tight', pad_inches=0,dpi=300)