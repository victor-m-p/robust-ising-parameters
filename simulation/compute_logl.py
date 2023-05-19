import numpy as np 
import re 
import os 
from sample_functions import read_text_file, ising_probs, bin_states, marginalize_n, logl
import pandas as pd 

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

#### LOGL #### 
data = np.loadtxt(f"{path_true}sim_true_nhid_0_nvis_{n_nodes}_th_gaussian_0.0_0.1_tj_gaussian_0.0_0.1_nsim_{n_sim}.txt")
configs = bin_states(n_nodes)
logl_true = logl(params = par_true, 
                 data = data, 
                 n_nodes = n_nodes, 
                 n_hidden = n_hidden, 
                 configs = configs)

n=100 
dct_hidden_logl = {key: [logl(dct_hidden[key][ele], data, n_nodes, n_hidden, configs) for ele in range(n)] for key in dct_hidden.keys()}

data_visible = data[:, n_hidden:]
configs_visible = bin_states(n_visible)
dct_visible_logl = {key: [logl(dct_visible[key][ele], data_visible, n_nodes-n_hidden, 0, configs_visible) for ele in range(n)] for key in dct_visible.keys()}

# create dataframe from this
def dct_to_df(dct, val):
    expanded = [(k, i+1, v_i) for k, v in dct.items() for i, v_i in enumerate(v)]
    df = pd.DataFrame(expanded, columns=['idx', 'num', f'{val}'])
    return df 

# save 
d_hidden = dct_to_df(dct_hidden_logl, 'logL')
d_visible = dct_to_df(dct_visible_logl, 'logL')

d_hidden.to_csv(f"data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/logL_hidden_n{n}.csv", index=False)
d_visible.to_csv(f"data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/logL_visible_n{n}.csv", index=False)

with open(f"data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/logL_true_n{n}.txt", "w") as logl_file:
    logl_file.write(str(logl_true))