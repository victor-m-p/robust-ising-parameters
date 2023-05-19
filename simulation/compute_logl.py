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
    logl_list = []
    for filename in files: 
        _, logl = read_text_file(f"{path}{filename}")
        logl_list.append(logl)
    return logl_list 

# load mpf data
files_hidden = [x for x in os.listdir(path_mpf) if x.endswith('_log.txt') and x.startswith(f'sim_hid_mpf_nhid_{n_hidden}')]
files_visible = [x for x in os.listdir(path_mpf) if x.endswith('_log.txt') and x.startswith('sim_hid_mpf_nhid_0')]

sparsity_regex = re.compile(r'(?<=txt_)(.*)(?<=_)')
sparsity_neg = np.arange(-1, 0.0, 0.05)
sparsity_neg = ["{:.2f}".format(num) for num in sparsity_neg]
sparsity_pos = ["{:05.2f}".format(i/100) for i in range(0, 105, 5)] # terrible formatting
sparsity_range = sparsity_neg + sparsity_pos

dct_logl_hidden = {}
dct_logl_visible = {}
for i in sparsity_range:
    files_hidden_i = [x for x in files_hidden if sparsity_regex.search(x).group(0).startswith(i)]
    files_visible_i = [x for x in files_visible if sparsity_regex.search(x).group(0).startswith(i)]
    
    logl_hidden = load_txt_dir(path_mpf, files_hidden_i)
    logl_visible = load_txt_dir(path_mpf, files_visible_i)

    dct_logl_hidden[i] = logl_hidden
    dct_logl_visible[i] = logl_visible

# load true params
filename = [x for x in os.listdir(path_true) if x.startswith('format')][0]
par_true = np.loadtxt(f"{path_true}{filename}")

# create dataframe from this
def dct_to_df(dct, val):
    expanded = [(k, i+1, v_i) for k, v in dct.items() for i, v_i in enumerate(v)]
    df = pd.DataFrame(expanded, columns=['idx', 'num', f'{val}'])
    return df 

# save 
d_hidden = dct_to_df(dct_logl_hidden, 'logL')
d_visible = dct_to_df(dct_logl_visible, 'logL')

d_hidden.to_csv(f"data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/logL_hidden_mpf.csv", index=False)
d_visible.to_csv(f"data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/logL_visible_mpf.csv", index=False)