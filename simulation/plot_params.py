import numpy as np 
import pandas as pd 
import re 
import os 
from sample_functions import read_text_file, regularization_penalty, param_magnitude
import matplotlib.pyplot as plt 
import seaborn as sns 
import arviz as az

# meta setup
n_nodes = 13
n_connections = int(n_nodes*(n_nodes-1)/2)
n_hidden = 3
n_visible = 10
n_obs = 500

# match the files
figpath = 'fig/fully_connected/'
path_mpf = f'data/fully_connected_nn{n_nodes}_nsim{n_obs}_mpf/'
path_true = f'data/fully_connected_nn{n_nodes}_nsim{n_obs}_true/'

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

# calculate the magnitude of paramters for models 
n_sim = 100
dct_magnitude_squared = {key: [param_magnitude(dct_par[key][ele], 2) for ele in range(n_sim)] for key in dct_par.keys()}
dct_magnitude_absolute = {key: [param_magnitude(dct_par[key][ele], 1) for ele in range(n_sim)] for key in dct_par.keys()}

# calculate the magnitude of parameters for true 
true_magnitude_squared = param_magnitude(par_true, 2)
true_magnitude_absolute = param_magnitude(par_true, 1) 

## calculate metrics for squared parameters 
hdi_prob = .95
mean_magnitude_squared = [(i, np.mean(dct_magnitude_squared[i])) for i in sparsity_range]
min_magnitude_squared = [(i, np.min(dct_magnitude_squared[i])) for i in sparsity_range]
HDI_magnitude_squared = [(i, az.hdi(np.array(dct_magnitude_squared[i]), hdi_prob=hdi_prob)) for i in sparsity_range]
HDI_magnitude_squared = [(t[0], t[1][0], t[1][1]) for t in HDI_magnitude_squared]

## gather metrics
df_mean = pd.DataFrame(mean_magnitude_squared, columns=['sparsity', 'mean_param_magnitude'])
df_min = pd.DataFrame(min_magnitude_squared, columns=['sparsity', 'min_param_magnitude'])
df_HDI = pd.DataFrame(HDI_magnitude_squared, columns=['sparsity', 'HDI_lower', 'HDI_upper'])
df_magnitude_squared = pd.merge(df_mean, df_HDI, on='sparsity')
df_magnitude_squared = pd.merge(df_magnitude_squared, df_min, on='sparsity')
df_magnitude_squared['sparsity'] = df_magnitude_squared['sparsity'].astype(float)

# main plot 
fig, ax = plt.subplots(figsize=(10, 6))

## Vertical line for each X from Y_lower to Y_upper
for _, row in df_magnitude_squared.iterrows():
    plt.plot([row['sparsity'], row['sparsity']], 
             [row['HDI_lower'], row['HDI_upper']], 
             color='tab:grey')

## Scatter plot for mean_Y
plt.scatter(df_magnitude_squared['sparsity'], 
            df_magnitude_squared['mean_param_magnitude'], 
            color='tab:blue', 
            label='mean(params**2)',
            zorder=2)

## Scatter plot for max_Y
plt.scatter(df_magnitude_squared['sparsity'], 
            df_magnitude_squared['min_param_magnitude'], 
            color='tab:orange', 
            label='min(params**2)',
            zorder=2)

## horizontal line for true 
plt.plot([-1, 1], [true_magnitude_squared, true_magnitude_squared], 
         color='tab:red', label='true params**2')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('mean(params**2) and min(params**2) with 95% HDI')
plt.tick_params(axis='x', rotation=45)
plt.xlabel('Sparsity')
plt.ylabel(r'$\sum \; \beta^2$')
plt.grid(True)
plt.legend()
plt.savefig(f"{figpath}param_magnitude_squared_nn{n_nodes}_nsim{n_sims}.png")

# plot distributions 
fig, ax = plt.subplots()
valrange=["-1.00", "-0.50", "00.00", "00.50"]
col=['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i in valrange:
    sns.histplot(dct_magnitude_squared[i], label=i, bins=10, color=col[valrange.index(i)])
plt.xlabel(r'$\sum \; \beta^2$')
plt.title('Distribution of params**2')
plt.legend()
plt.savefig(f"{figpath}param_magnitude_squared_nn{n_nodes}_nsim{n_sims}_distributions.png")
plt.close()