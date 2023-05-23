'''
Last step takes a long time for large n (e.g. n > 15). 
'''

import numpy as np 
import re 
import os 
from sample_functions import read_text_file, logl, check_valid
import pandas as pd 

# meta setup
n_nodes = 21
n_hidden = 1
n_connections = int(n_nodes*(n_nodes-1)/2)
n_visible = n_nodes-n_hidden
n_sim = 500
norm = 'l1' # l2 
condition = 'not_connected' # fully_connected

outpath = f"data/{condition}_nn{n_nodes}_nsim{n_sim}_{norm}_params/"
if not os.path.exists(outpath): 
    os.makedirs(outpath)

# match the files
path_mpf = f'data/{condition}_nn{n_nodes}_nsim{n_sim}_{norm}_mpf/'
path_true = f'data/{condition}_nn{n_nodes}_nsim{n_sim}_true/'

# load files helper  
def load_txt_dir(path, files):
    logl_list = []
    for filename in files: 
        _, logl = read_text_file(f"{path}{filename}")
        logl_list.append(logl)
    return logl_list 

# changed format 
files_hidden = [x for x in os.listdir(path_mpf) if x.endswith('_log.txt') and x.startswith(f'sim_mpf_nhid_{n_hidden}')]
files_visible = [x for x in os.listdir(path_mpf) if x.endswith('_log.txt') and x.startswith('sim_mpf_nhid_0')]
sparsity_regex = re.compile(r'(?<=txt_)(.*)(?<=_)')
sparsity_neg = np.arange(-1, 0.0, 0.1)
sparsity_neg = ["{:.1f}".format(num) for num in sparsity_neg] # ["{:.2f}".format(num) for num in sparsity_neg]
sparsity_pos = ["{:04.1f}".format(i/100) for i in range(0, 105, 10)] # ["{:05.2f}".format(i/100) for i in range(0, 105, 5)]
sparsity_range = sparsity_neg + sparsity_pos

dct_hidden = {}
dct_visible = {}
for i in sparsity_range:
    files_hidden_i = [x for x in files_hidden if sparsity_regex.search(x).group(0).startswith(i)]
    files_visible_i = [x for x in files_visible if sparsity_regex.search(x).group(0).startswith(i)]
    
    logl_hidden = load_txt_dir(path_mpf, files_hidden_i)
    logl_visible = load_txt_dir(path_mpf, files_visible_i)

    dct_hidden[i] = logl_hidden
    dct_visible[i] = logl_visible

# delete empty elements (i.e., if run over smaller grid)
dct_hidden = {k: v for k, v in dct_hidden.items() if check_valid(v)}
dct_visible = {k: v for k, v in dct_visible.items() if check_valid(v)}

# create dataframe from this
def dct_to_df(dct, val):
    expanded = [(k, i+1, v_i) for k, v in dct.items() for i, v_i in enumerate(v)]
    df = pd.DataFrame(expanded, columns=['idx', 'num', f'{val}'])
    return df 

# save the mpf logl  
d_hidden = dct_to_df(dct_hidden, 'logL')
d_visible = dct_to_df(dct_visible, 'logL')

d_hidden.to_csv(f"{outpath}logL_hidden_mpf.csv", index=False)
d_visible.to_csv(f"{outpath}logL_visible_mpf.csv", index=False)

# load true params
filename = [x for x in os.listdir(path_true) if x.startswith('format')][0]
par_true = np.loadtxt(f"{path_true}{filename}")

# compute true logl 
true_data_path = [x for x in os.listdir(path_true) if x.startswith('sim_true')][0] 
true_data = np.loadtxt(f"{path_true}{true_data_path}")
logl_true = logl(par_true, true_data, n_nodes, n_hidden)

with open(f'{outpath}logL_true.txt', 'w') as f: 
    f.write(str(logl_true))
    