import numpy as np 
import re 
import os 
from sample_functions import read_text_file, ising_probs, bin_states, marginalize_n
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

## NEW: save params without 
with open(f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/DKL_hidden_n{n}.json', 'w') as f: 
    json.dump(dct_hidden_DKL, f)

with open(f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/DKL_visible_n{n}.json', 'w') as f:
    json.dump(dct_visible_DKL, f)

'''
def construct_DKL_df(dct_DKL, sparsity_range): 
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
    return df_DKL 

df_DKL_visible = construct_DKL_df(dct_visible_DKL, sparsity_range)
df_DKL_hidden = construct_DKL_df(dct_hidden_DKL, sparsity_range)

# save 
param_dir = f"data/fully_connected_nn{n_nodes}_nsim{n_sim}_params/"
os.mkdir(f"{param_dir}")
df_DKL_visible.to_csv(f"{param_dir}DKL_visible_n100.csv")
df_DKL_hidden.to_csv(f"{param_dir}DKL_hidden_n100.csv")

'''


