import numpy as np 
import pandas as pd 
import re 
import os 
from sample_functions import read_text_file, param_magnitude_mean

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

# load mpf data
files_hidden = [x for x in os.listdir(path_mpf) if x.endswith('_log.txt') and x.startswith(f'sim_hid_mpf_nhid_{n_hidden}')]
files_visible = [x for x in os.listdir(path_mpf) if x.endswith('_log.txt') and x.startswith('sim_hid_mpf_nhid_0')]

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
    
    params_hidden = load_txt_dir(path_mpf, files_hidden_i)
    params_visible = load_txt_dir(path_mpf, files_visible_i)

    dct_hidden[i] = params_hidden
    dct_visible[i] = params_visible 

# load true params
filename = [x for x in os.listdir(path_true) if x.startswith('format')][0]
par_true = np.loadtxt(f"{path_true}{filename}")

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

# calculate the magnitude of paramters for models 
dct_magnitude_hidden = {key: [param_magnitude_mean(dct_obspar_hidden[key][ele], 2) for ele in range(len(dct_hidden['-1.00']))] for key in dct_obspar_hidden.keys()}
dct_magnitude_visible = {key: [param_magnitude_mean(dct_visible[key][ele], 2) for ele in range(len(dct_hidden['-1.00']))] for key in dct_visible.keys()}

def dct_to_df(dct, val):
    expanded = [(k, i+1, v_i) for k, v in dct.items() for i, v_i in enumerate(v)]
    df = pd.DataFrame(expanded, columns=['idx', 'num', f'{val}'])
    return df 

df_magnitude_hidden = dct_to_df(dct_magnitude_hidden, 'squared_magnitude')
df_magnitude_visible = dct_to_df(dct_magnitude_visible, 'squared_magnitude')

df_magnitude_hidden.to_csv(f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/param_magnitude_hidden.csv', index=False)
df_magnitude_visible.to_csv(f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/param_magnitude_visible.csv', index=False)

magnitude_true = param_magnitude_mean(obspar_true, 2)
with open(f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/param_magnitude_true.txt', 'w') as f: 
    f.write(str(magnitude_true))

## calculate average error ## 
def calculate_param_MSE(dct_observed, dct_true, sparsity_range):
    dct_error = {}
    for i in sparsity_range: 
        dct_error[i] = [np.mean((dct_observed[i][ele] - dct_true)**2) for ele in range(len(dct_observed[i]))]
    return dct_error

dct_obspar_hidden_MSE = calculate_param_MSE(dct_obspar_hidden, obspar_true, sparsity_range)
dct_obspar_visible_MSE = calculate_param_MSE(dct_visible, obspar_true, sparsity_range)

df_MSE_hidden = dct_to_df(dct_obspar_hidden_MSE, 'MSE')
df_MSE_visible = dct_to_df(dct_obspar_visible_MSE, 'MSE')

df_MSE_hidden.to_csv(f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/param_MSE_hidden.csv', index=False)
df_MSE_visible.to_csv(f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/param_MSE_visible.csv', index=False)