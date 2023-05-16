import numpy as np 
import pandas as pd 
import re 
import os 
from sample_functions import read_text_file, plot_params, plot_h_hidden, ising_probs, bin_states, marginalize_n, deconstruct_J, DKL
import matplotlib.pyplot as plt 
import seaborn as sns 
import arviz as az

# meta setup
n_nodes = 13
n_hidden = 3
n_connections = int(n_nodes*(n_nodes-1)/2)
n_visible = n_nodes-n_hidden
n_sim = 500

# match the files
figpath = 'fig/fully_connected/'
path_mpf = f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_mpf/'
path_true = f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_true/'

# load files helper  
def load_txt_dir(path, files):
    par_list = []
    for filename in files: 
        params, _ = read_text_file(f"{path}{filename}")
        par_list.append(params)
    return par_list

# load data
files_hidden = [x for x in os.listdir(path_mpf) if x.endswith('_log.txt') and x.startswith('sim_hid')]
sparsity_regex = re.compile(r'(?<=txt_)(.*)(?<=_)')
sparsity_neg = np.arange(-1, 0.0, 0.05)
sparsity_neg = ["{:.2f}".format(num) for num in sparsity_neg]
sparsity_pos = ["{:05.2f}".format(i/100) for i in range(0, 105, 5)] # terrible formatting
sparsity_range = sparsity_neg + sparsity_pos

dct_par = {}
for i in sparsity_range:
    files = [x for x in files_hidden if sparsity_regex.search(x).group(0).startswith(i)]
    par = load_txt_dir(path_mpf, files)
    dct_par[i] = par

# load true params
filename = [x for x in os.listdir(path_true) if x.startswith('format')][0]
par_true = np.loadtxt(f"{path_true}{filename}")

# faster DKL where we do not recompute 
# see DKL function in sample_functions.py  
def DKL_precompute(inverse,
                   params_model, 
                   true_probs_marginal, 
                   n_nodes):
    
    # get model probabilities
    nj = int(n_nodes*(n_nodes-1)/2)
    h_model = params_model[nj:]
    J_model = params_model[:nj]
    model_probs = ising_probs(h_model, J_model)
    
    # quick marginalize 
    model_probs_marginal = np.bincount(inverse, weights=model_probs)
    return np.sum(true_probs_marginal*np.log(true_probs_marginal/model_probs_marginal))

# precompute to speed up 
configs = bin_states(n_nodes)
true_probs = ising_probs(par_true[n_connections:], par_true[:n_connections])
_, true_probs_marginal = marginalize_n(configs, true_probs, n_hidden)

reduced_configurations = configs[:, n_hidden:]
_, inverse = np.unique(reduced_configurations, axis=0, return_inverse=True)

# calculate DKL 
n_sim = 100 # already takes a little while 
dct_DKL = {key: [DKL_precompute(inverse, dct_par[key][ele], true_probs_marginal, n_nodes) for ele in range(n_sim)] for key in dct_par.keys()}

## calculate metrics 
hdi_prob = .95
mean_DKL = [(i, np.mean(dct_DKL[i])) for i in sparsity_range]
min_DKL = [(i, np.min(dct_DKL[i])) for i in sparsity_range]
HDI_DKL = [(i, az.hdi(np.array(dct_DKL[i]), hdi_prob=hdi_prob)) for i in sparsity_range]
HDI_DKL = [(t[0], t[1][0], t[1][1]) for t in HDI_DKL]
## gather metrics
df_mean = pd.DataFrame(mean_DKL, columns=['sparsity', 'mean_DKL'])
df_min = pd.DataFrame(min_DKL, columns=['sparsity', 'min_DKL'])
df_HDI = pd.DataFrame(HDI_DKL, columns=['sparsity', 'HDI_lower', 'HDI_upper'])
df_DKL = pd.merge(df_mean, df_HDI, on='sparsity')
df_DKL = pd.merge(df_DKL, df_min, on='sparsity')

# plot mean(HDI) and min(HDI)
fig, ax = plt.subplots(figsize=(10, 6))

# Vertical line for each X from Y_lower to Y_upper
for _, row in df_DKL.iterrows():
    plt.plot([row['sparsity'], row['sparsity']], 
             [row['HDI_lower'], row['HDI_upper']], 
             color='tab:grey')

# Scatter plot for mean_Y
plt.scatter(df_DKL['sparsity'], 
            df_DKL['mean_DKL'], 
            color='tab:blue', 
            label='mean(DKL)',
            zorder=2)

# Scatter plot for max_Y
plt.scatter(df_DKL['sparsity'], 
            df_DKL['min_DKL'], 
            color='tab:orange', 
            label='min(DKL)',
            zorder=2)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('mean(HDI) and min(HDI) with 95% HDI')
plt.tick_params(axis='x', rotation=45)
plt.xlabel('Sparsity')
plt.ylabel(r'$D_{KL}(P_{true}||P_{model})$')
plt.grid(True)
plt.legend()
plt.savefig(f"{figpath}DKL_L2_nn{n_nodes}_nsim{n_sim}_overview.png")
plt.close()

# plot distributions 
fig, ax = plt.subplots()
valrange=["-1.00", "-0.50", "00.00", "00.50"]
col=['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i in valrange:
    sns.histplot(dct_DKL[i], label=i, bins=10, color=col[valrange.index(i)])
plt.xlabel(r'$D_{KL}(P_{true}||P_{model})$')
plt.title('Distribution of DKL')
plt.legend()
plt.savefig(f"{figpath}DKL_L2_nn{n_nodes}_nsim{n_sim}_distributions.png")
plt.close()

###### delete the below ########


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
