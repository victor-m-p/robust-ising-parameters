import numpy as np 
import re 
import os 
from sample_functions import read_text_file, ising_probs, bin_states, marginalize_n
import pandas as pd 

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
path_mpf = f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_l1_mpf/'
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

###### DKL ######

# fastest DKL I have been able to make currently 
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

def DKL_visible(params_model,
                true_probs_marginal,
                n_nodes):
    
    # get model probabilities
    nj = int(n_nodes*(n_nodes-1)/2)
    h_model = params_model[nj:]
    J_model = params_model[:nj]
    model_probs = ising_probs(h_model, J_model)
    
    return np.sum(true_probs_marginal*np.log(true_probs_marginal/model_probs))

# precompute to speed up 
configs = bin_states(n_nodes)
true_probs = ising_probs(par_true[n_connections:], par_true[:n_connections])
_, true_probs_marginal = marginalize_n(configs, true_probs, n_hidden)

reduced_configurations = configs[:, n_hidden:]
_, inverse = np.unique(reduced_configurations, axis=0, return_inverse=True)

# calculate DKL hidden
n = 100 # already takes a little while 
dct_hidden_DKL = {key: [DKL_precompute(inverse, dct_hidden[key][ele], true_probs_marginal, n_nodes) for ele in range(n)] for key in dct_hidden.keys()}
dct_visible_DKL = {key: [DKL_visible(dct_visible[key][ele], true_probs_marginal, n_visible) for ele in range(n)] for key in dct_visible.keys()}

# create dataframe from this
def dct_to_df(dct, val):
    expanded = [(k, i+1, v_i) for k, v in dct.items() for i, v_i in enumerate(v)]
    df = pd.DataFrame(expanded, columns=['idx', 'num', f'{val}'])
    return df 

## NEW: save params without 
d_hidden = dct_to_df(dct_hidden_DKL, 'DKL')
d_visible = dct_to_df(dct_visible_DKL, 'DKL')

d_hidden.to_csv(f"{outpath}DKL_hidden_n{n}.csv", index=False)
d_visible.to_csv(f"{outpath}DKL_visible_n{n}.csv", index=False)