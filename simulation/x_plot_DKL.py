import numpy as np 
import pandas as pd 
import re 
import os 
from sample_functions import read_text_file, plot_params, plot_h_hidden, ising_probs, bin_states, marginalize_n, deconstruct_J
import matplotlib.pyplot as plt 
import seaborn as sns 

# meta setup
n_nodes = 11
n_connections = int(n_nodes*(n_nodes-1)/2)
n_hidden = 3
n_visible = 8

# match the files
figpath = 'fig/fully_connected/'
path_mpf = 'data/fully_connected_big_grid/'
path_true = 'data/fully_connected_true_big/'

# load files helper  
def load_txt_dir(path, files, n_connections):
    h_list = []
    J_list = []
    logl_list = []
    par_list = []
    for filename in files: 
        params, logl = read_text_file(f"{path}{filename}")
        par_list.append(params)
        logl_list.append(logl)
        h_list.append(params[n_connections:])
        J_list.append(params[:n_connections])
    return par_list, h_list, J_list, logl_list

# for different lambda (sparsity) values
files_hidden = [x for x in os.listdir(path_mpf) if x.endswith('_log.txt') and x.startswith('sim_hid')]
sparsity_regex = re.compile(r'(?<=txt_)(.*)(?<=_)')
sparsity_neg = np.arange(-1, 0.0, 0.05)
sparsity_neg = ["{:.2f}".format(num) for num in sparsity_neg]
# the positive numbers are formatted in a terrible way apparently: 
sparsity_pos = ["00.00", "00.05", "00.10", "00.15", "00.20", "00.25", "00.30", "00.35", "00.40"]
sparsity_range = sparsity_neg + sparsity_pos

dct_par = {}
dct_h = {}
dct_J = {}
dct_logl = {}
for i in sparsity_range:
    files = [x for x in files_hidden if sparsity_regex.search(x).group(0).startswith(i)]
    par, h, J, logl = load_txt_dir(path_mpf, files, n_connections)
    dct_par[i] = par
    dct_h[i] = h
    dct_J[i] = J
    dct_logl[i] = logl

# sanity check that we have the same amount for each
n_ele = [len(dct_h[i]) for i in sparsity_range] # looks good

## log likelihood ## 
# first plot mean vs max loglikelihood
mean_logl = [(i, np.mean(dct_logl[i])) for i in sparsity_range]
max_logl = [(i, np.max(dct_logl[i])) for i in sparsity_range]
df_mean = pd.DataFrame(mean_logl, columns=['sparsity', 'mean_logl'])
df_max = pd.DataFrame(max_logl, columns=['sparsity', 'max_logl'])
df_logl = pd.merge(df_mean, df_max, on='sparsity')

# plot this 
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(df_logl['sparsity'], df_logl['mean_logl'], label='mean', color='tab:blue')
ax1.set_xlabel('sparsity')
ax1.set_ylabel('mean logl', color='tab:blue')
ax1.tick_params('y', colors='tab:blue')
ax1.tick_params(axis='x', rotation=45)
ax2 = ax1.twinx()
ax2.plot(df_logl['sparsity'], df_logl['max_logl'], label='max', color='tab:orange')
ax2.set_ylabel('max logl', color='tab:orange')
ax2.tick_params('y', colors='tab:orange')
fig.tight_layout()
plt.show();

# distributions 
valrange=["-1.00", "-0.50", "00.00", "00.40"]
fig, ax = plt.subplots(figsize=(7, 4))
for i in valrange:
    sns.kdeplot(dct_logl[i], label=i)
plt.xlabel('logl')
plt.legend()
plt.show();

## penalty ## 
norm=2.0
def param_norm(params, norm): 
    return np.sum(np.abs(params**norm))

# get the penalty for each parameter set 
par_penalty = {k: [param_norm(x, norm) for x in v] for k, v in dct_par.items()}

# get the true penalty 
filename = [x for x in os.listdir(path_true) if x.startswith('format')][0]
true_par = np.loadtxt(f"{path_true}{filename}")
true_penalty = param_norm(true_par, norm)

# plot distributions first 
fig, ax = plt.subplots(figsize=(7,4))
for i in valrange: 
    sns.kdeplot(par_penalty[i], label=i)
plt.vlines(true_penalty, 
           ymin=0,
           ymax=5,
           label='true', 
           linestyles='--',
           color='black')
plt.xlabel('sum(params**norm), norm=2')
plt.legend()
plt.show();

fig, ax = plt.subplots(figsize=(7,4))
sns.kdeplot(par_penalty["00.40"], label="00.40")
plt.vlines(true_penalty, 
           ymin=0,
           ymax=5,
           label='true', 
           linestyles='--',
           color='black')
plt.xlabel('sum(params**norm), norm=2')
plt.legend()
plt.show();

# are small params good params?
fig, axs = plt.subplots(1, 3, figsize=(15, 4))  # 1 row, 3 columns

# First plot
i = "00.40"
axs[0].scatter(par_penalty[i], dct_logl[i])
axs[0].set_title('sparsity=0.4')
axs[0].set_xlabel('sum(params**norm), norm=2')

# Second plot
i = "00.00"
axs[1].scatter(par_penalty[i], dct_logl[i])
axs[1].set_title('sparsity=0.0')
axs[1].set_xlabel('sum(params**norm), norm=2')

# Third plot
i = "-0.50"
axs[2].scatter(par_penalty[i], dct_logl[i])
axs[2].set_title('sparsity=-0.5')
axs[2].set_xlabel('sum(params**norm), norm=2')

plt.show()
