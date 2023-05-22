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
norm = 'l1'

# create directory if does not exist
outpath = f"data/fully_connected_nn{n_nodes}_nsim{n_sim}_{norm}_params/"
if not os.path.exists(outpath): 
    os.makedirs(outpath)

# match the files
path_mpf = f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_{norm}_mpf/'
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

### magnitude on all params ###
dct_magnitude_hidden = {key: [param_magnitude_mean(dct_hidden[key][ele], 2) for ele in range(len(dct_hidden['-1.00']))] for key in dct_hidden.keys()}
dct_magnitude_visible = {key: [param_magnitude_mean(dct_visible[key][ele], 2) for ele in range(len(dct_hidden['-1.00']))] for key in dct_visible.keys()}

dct_magnitude_hidden
dct_magnitude_visible

import matplotlib.pyplot as plt 
plt.plot(dct_magnitude_hidden['-0.30'])
plt.plot(dct_magnitude_visible['-0.30'])

par_hid = dct_hidden['-0.30'][0]

def extract_params(params, n_nodes, n_hidden, type='visible'): 
    # extract h, J
    n_connections = int(n_nodes*(n_nodes-1)/2)
    h = params[n_connections:]
    J = params[:n_connections]
    # observed 
    n_visible = n_nodes - n_hidden
    n_visible_connections = int(n_visible*(n_visible-1)/2)
    n_total_connections = int(n_nodes*(n_nodes-1)/2)
    n_hidden_connections = n_total_connections - n_visible_connections
    if type == 'hidden':
        h_sub = h[:n_hidden] 
        J_sub = J[:n_hidden_connections] 
    elif type == 'visible': 
        h_sub = h[n_hidden:] 
        J_sub = J[n_hidden_connections:]
    else: 
        print('type must be hidden or visible')
    params_sub = np.concatenate((J_sub, h_sub))
    return params_sub 

# extract params
h_vis = extract_params(par_hid, n_nodes, n_hidden, type='visible')
t_vis = extract_params(par_true, n_nodes, n_hidden, type='visible')
plt.scatter(h_vis, t_vis)

h_hid = extract_params(par_hid, n_nodes, n_hidden, type='hidden')
t_hid = extract_params(par_true, n_nodes, n_hidden, type='hidden')
plt.scatter(h_hid, t_hid)