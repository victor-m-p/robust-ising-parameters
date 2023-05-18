import numpy as np 
import pandas as pd 
import re 
import os 
from sample_functions import read_text_file, param_magnitude_mean
import matplotlib.pyplot as plt 
import seaborn as sns 
import arviz as az
import json

# meta setup
n_nodes = 13
n_hidden = 3
n_connections = int(n_nodes*(n_nodes-1)/2)
n_visible = n_nodes-n_hidden
n_sim = 500

# match the files
figpath = 'fig/fully_connected/'
path_mpf_hid = f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_mpf/'
path_mpf_vis = f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_visible_mpf/'
path_true = f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_true/'

# load files helper  
def load_txt_dir(path, files):
    par_list = []
    for filename in files: 
        params, _ = read_text_file(f"{path}{filename}")
        par_list.append(params)
    return par_list

# load mpf data
files_hidden = [x for x in os.listdir(path_mpf_hid) if x.endswith('_log.txt') and x.startswith('sim_hid')]
files_visible = [x for x in os.listdir(path_mpf_vis) if x.endswith('_log.txt') and x.startswith('sim_hid')]

sparsity_regex = re.compile(r'(?<=txt_)(.*)(?<=_)')
sparsity_neg = np.arange(-1, 0.0, 0.05)
sparsity_neg = ["{:.2f}".format(num) for num in sparsity_neg]
sparsity_pos = ["{:05.2f}".format(i/100) for i in range(0, 105, 5)] # terrible formatting
sparsity_range = sparsity_neg + sparsity_pos

dct_hidden = {}
dct_visible = {}
for i in sparsity_range:
    files_hidden_i = [x for x in files_hidden if sparsity_regex.search(x).group(0).startswith(i)]
    files_visible_i = [x for x in files_visible if sparsity_regex.search(x).group(0).startswith(i)]
    
    params_hidden = load_txt_dir(path_mpf_hid, files_hidden_i)
    params_visible = load_txt_dir(path_mpf_vis, files_visible_i)

    dct_hidden[i] = params_hidden
    dct_visible[i] = params_visible 

# load true params
filename = [x for x in os.listdir(path_true) if x.startswith('format')][0]
par_true = np.loadtxt(f"{path_true}{filename}")

### compare magnitude of parameters ###

# calculate the magnitude of paramters for models 
dct_magnitude_hidden = {key: [param_magnitude_mean(dct_hidden[key][ele], 2) for ele in range(len(dct_hidden['-1.00']))] for key in dct_hidden.keys()}
dct_magnitude_visible = {key: [param_magnitude_mean(dct_visible[key][ele], 2) for ele in range(len(dct_hidden['-1.00']))] for key in dct_visible.keys()}

### should also be for only the visible ... ### 

#with open('../data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/magnitude_observed_hidden.json', 'w') as f: 
#    json.dump(dct_magnitude_hidden, f)

#with open('../data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/magnitude_observed_visible.json', 'w') as f:
#    json.dump(dct_magnitude_visible, f)

# calculate the magnitude of parameters for true 
true_magnitude = param_magnitude_mean(par_true, 2)

def construct_par_df(dct_magnitude, sparsity_range): 
    ## calculate metrics for squared parameters 
    hdi_prob = .95
    mean_magnitude = [(i, np.mean(dct_magnitude[i])) for i in sparsity_range]
    min_magnitude = [(i, np.min(dct_magnitude[i])) for i in sparsity_range]
    HDI_magnitude = [(i, az.hdi(np.array(dct_magnitude[i]), hdi_prob=hdi_prob)) for i in sparsity_range]
    HDI_magnitude = [(t[0], t[1][0], t[1][1]) for t in HDI_magnitude]

    ## gather metrics
    df_mean = pd.DataFrame(mean_magnitude, columns=['sparsity', 'mean_param_magnitude'])
    df_min = pd.DataFrame(min_magnitude, columns=['sparsity', 'min_param_magnitude'])
    df_HDI = pd.DataFrame(HDI_magnitude, columns=['sparsity', 'HDI_lower', 'HDI_upper'])
    df_magnitude = pd.merge(df_mean, df_HDI, on='sparsity')
    df_magnitude = pd.merge(df_magnitude, df_min, on='sparsity')
    df_magnitude['sparsity'] = df_magnitude['sparsity'].astype(float)
    
    return df_magnitude 

df_magnitude_hidden = construct_par_df(dct_magnitude_hidden, sparsity_range)
df_magnitude_visible = construct_par_df(dct_magnitude_visible, sparsity_range)

# overview of the magnitude of parameters
def plot_param_overview(df_magnitude, true_magnitude, n_nodes, n_sim, condition): 

    # main plot 
    fig, ax = plt.subplots(figsize=(10, 6))

    ## Vertical line for each X from Y_lower to Y_upper
    for _, row in df_magnitude.iterrows():
        plt.plot([row['sparsity'], row['sparsity']], 
                [row['HDI_lower'], row['HDI_upper']], 
                color='tab:grey')

    ## Scatter plot for mean_Y
    plt.scatter(df_magnitude['sparsity'], 
                df_magnitude['mean_param_magnitude'], 
                color='tab:blue', 
                label='mean(params**2)',
                zorder=2)

    ## Scatter plot for max_Y
    plt.scatter(df_magnitude['sparsity'], 
                df_magnitude['min_param_magnitude'], 
                color='tab:orange', 
                label='min(params**2)',
                zorder=2)

    ## horizontal line for true 
    plt.plot([-1, 1], [true_magnitude, true_magnitude], 
            color='tab:red', label='true params**2')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('mean(params**2) and min(params**2) with 95% HDI')
    plt.tick_params(axis='x', rotation=45)
    plt.xlabel('Sparsity')
    plt.ylabel(r'$\sum \; \beta^2$')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{figpath}param_magnitude_nn{n_nodes}_nsim{n_sim}_{condition}.png")

plot_param_overview(df_magnitude_hidden, true_magnitude, n_nodes, n_sim, 'hidden')
plot_param_overview(df_magnitude_visible, true_magnitude, n_nodes, n_sim, 'visible')

## compare actual parameters ## 
def observed_params(params, n_nodes, n_hidden): 
    # extract h, J
    n_connections = int(n_nodes*(n_nodes-1)/2)
    h = params[n_connections:]
    J = params[:n_connections]
    # observed 
    n_visible = n_nodes - n_hidden
    h_observed = h[n_hidden:] # last n_visible 
    J_observed = J[-n_visible*(n_visible-1)//2:] # last n_visible*(n_visible-1)/2
    params_observed = np.concatenate((J_observed, h_observed))
    return params_observed 

obspar_true = observed_params(par_true, n_nodes, n_hidden)
dct_obspar_hidden = {key: [observed_params(dct_hidden[key][ele], n_nodes, n_hidden) for ele in range(len(dct_hidden['-1.00']))] for key in dct_hidden.keys()}

## calculate average error ## 
def calculate_param_MSE(dct_observed, dct_true, sparsity_range):
    dct_error = {}
    for i in sparsity_range: 
        dct_error[i] = [np.mean((dct_observed[i][ele] - dct_true)**2) for ele in range(len(dct_observed[i]))]
    return dct_error

dct_obspar_hidden_MSE = calculate_param_MSE(dct_obspar_hidden, obspar_true, sparsity_range)
dct_obspar_visible_MSE = calculate_param_MSE(dct_visible, obspar_true, sparsity_range)

## NEW: save params without 
with open(f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/MSE_observed_hidden.json', 'w') as f: 
    json.dump(dct_obspar_hidden_MSE, f)

with open(f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/MSE_observed_visible.json', 'w') as f:
    json.dump(dct_obspar_visible_MSE, f)

## plot average parameter error ##
df_error_hidden = construct_par_df(dct_obspar_hidden_MSE, sparsity_range)
df_error_visible = construct_par_df(dct_obspar_visible_MSE, sparsity_range)

def plot_param_error(df_error, n_nodes, n_sim, condition):
    
    # main plot 
    fig, ax = plt.subplots(figsize=(10, 6))

    ## Vertical line for each X from Y_lower to Y_upper
    for _, row in df_error.iterrows():
        plt.plot([row['sparsity'], row['sparsity']], 
                [row['HDI_lower'], row['HDI_upper']], 
                color='tab:grey')

    ## Scatter plot for mean_Y
    plt.scatter(df_error['sparsity'], 
                df_error['mean_param_magnitude'], 
                color='tab:blue', 
                label='mean(MSE(params))',
                zorder=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('mean(MSE(params)) with 95% HDI')
    plt.tick_params(axis='x', rotation=45)
    plt.xlabel('Sparsity')
    plt.ylabel(r'$\sum \; \beta^2$')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{figpath}MSE_observed_params_nn{n_nodes}_nsim{n_sim}_{condition}.png")

plot_param_error(df_error_hidden, n_nodes, n_sim, 'hidden')
plot_param_error(df_error_visible, n_nodes, n_sim, 'visible')
