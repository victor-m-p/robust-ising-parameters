import numpy as np 
import pandas as pd 
import re 
import os 
from sample_functions import read_text_file, plot_params, plot_h_hidden, ising_probs, bin_states, marginalize_n, deconstruct_J
import matplotlib.pyplot as plt 
import seaborn as sns 

# meta setup
n_nodes = 11
n_connections = int(n_nodes*(n_nodes-1)/2)
n_hidden = 3
n_visible = 8

# match the files
figpath = 'fig/fully_connected/'
path_mpf = 'data/fully_connected_grid/'
path_true = 'data/fully_connected_true_big/'

# load files helper  
def load_txt_dir(path, files, n_connections):
    h_list = []
    J_list = []
    logl_list = []
    for filename in files: 
        params, logl = read_text_file(f"{path}{filename}")
        logl_list.append(logl)
        h_list.append(params[n_connections:])
        J_list.append(params[:n_connections])
    return h_list, J_list, logl_list

# for different lambda (sparsity) values
files_hidden = [x for x in os.listdir(path_mpf) if x.endswith('_log.txt') and x.startswith('sim_hid')]
sparsity_regex = re.compile(r'(?<=txt_)(.*)(?<=_)')

# -2.0
files_neg2 = [x for x in files_hidden if sparsity_regex.search(x).group(0).startswith('-2.0')]
h_hidden_neg2, J_hidden_neg2, logl_hidden_neg2 = load_txt_dir(path_mpf, files_neg2, n_connections)

# -1.0
files_neg1 = [x for x in files_hidden if sparsity_regex.search(x).group(0).startswith('-1.0')]
h_hidden_neg1, J_hidden_neg1, logl_hidden_neg1 = load_txt_dir(path_mpf, files_neg1, n_connections)

# 0.0
files_0 = [x for x in files_hidden if sparsity_regex.search(x).group(0).startswith('0.0')]
h_hidden_0, J_hidden_0, logl_hidden_0 = load_txt_dir(path_mpf, files_0, n_connections)

# 1.0
files_1 = [x for x in files_hidden if sparsity_regex.search(x).group(0).startswith('1.0')]
h_hidden_1, J_hidden_1, logl_hidden_1 = load_txt_dir(path_mpf, files_1, n_connections)

# 2.0 
files_2 = [x for x in files_hidden if sparsity_regex.search(x).group(0).startswith('2.0')]
h_hidden_2, J_hidden_2, logl_hidden_2 = load_txt_dir(path_mpf, files_2, n_connections)

# best average logl?
np.mean(logl_hidden_neg2) # -21946.20
np.mean(logl_hidden_neg1) # -21634.06
np.mean(logl_hidden_0) # -21453.58
np.mean(logl_hidden_1) # -21525.36
np.mean(logl_hidden_2) # -22048.44 

# best absolute logl?
np.max(logl_hidden_neg2) # -21324.89
np.max(logl_hidden_neg1) # -21242.38
np.max(logl_hidden_0) # -21252.27
np.max(logl_hidden_1) # -21525.36
np.max(logl_hidden_2) # -22048.44

# variance 
np.std(logl_hidden_neg2) # 555.13
np.std(logl_hidden_neg1) # 181.60
np.std(logl_hidden_0) # 114.12
np.std(logl_hidden_1) # 4.17e-05
np.std(logl_hidden_2) # 5.28e-06

# best for each case
best_logl_idx_neg2 = np.where(logl_hidden_neg2 == np.max(logl_hidden_neg2))[0][0]
best_logl_idx_neg1 = np.where(logl_hidden_neg1 == np.max(logl_hidden_neg1))[0][0]
best_logl_idx_0 = np.where(logl_hidden_0 == np.max(logl_hidden_0))[0][0]
best_logl_idx_1 = np.where(logl_hidden_1 == np.max(logl_hidden_1))[0][0]
best_logl_idx_2 = np.where(logl_hidden_2 == np.max(logl_hidden_2))[0][0]

# 556: 
# best fit could 
np.log(0.67)
# read actual shit 
param_files = [x for x in os.listdir(path_true) if x.startswith('format')]
filename = param_files[0]
params_true = np.loadtxt(f'{path_true}{filename}')
h_true = params_true[n_connections:]
J_true = params_true[:n_connections]

np.sum(params_true**2) # 0.6715
hbest = h_hidden_1[best_logl_idx_1]
jbest = J_hidden_1[best_logl_idx_1]
hJbest = np.concatenate((hbest, jbest))
np.sum(hJbest**2) # 67.15 

# compare h (hmmm)
n_hidden = 3
plot_h_hidden(h_true, h_hidden_neg2[best_logl_idx_neg2], n_hidden, 'x', 0.1)
plot_h_hidden(h_true, h_hidden_neg1[best_logl_idx_neg1], n_hidden, 'x', 0.1)
plot_h_hidden(h_true, h_hidden_0[best_logl_idx_0], n_hidden, 'x', 0.1)
plot_h_hidden(h_true, h_hidden_1[best_logl_idx_1], n_hidden, 'x', 0.1) # nonsense
plot_h_hidden(h_true, h_hidden_2[best_logl_idx_2], n_hidden, 'x', 0.1) # nonsense

# compare J 
plot_params(J_true, J_hidden_neg2[best_logl_idx_neg2], 'x', 0.1) # crazy
plot_params(J_true, J_hidden_neg1[best_logl_idx_neg1], 'x', 0.1) # crazy
plot_params(J_true, J_hidden_0[best_logl_idx_0], 'x', 0.1)
plot_params(J_true, J_hidden_1[best_logl_idx_1], 'x', 0.1) # sort-of nonsense. 
plot_params(J_true, J_hidden_2[best_logl_idx_2], 'x', 0.1) # just completely flat 

# compare p states 
configurations = bin_states(n_nodes)
true_probs = ising_probs(h_true, J_true)
hid_probs_neg2 = ising_probs(h_hidden_neg2[best_logl_idx_neg2], 
                             J_hidden_neg2[best_logl_idx_neg2])
hid_probs_neg1 = ising_probs(h_hidden_neg1[best_logl_idx_neg1],
                             J_hidden_neg1[best_logl_idx_neg1])
hid_probs_0 = ising_probs(h_hidden_0[best_logl_idx_0],
                          J_hidden_0[best_logl_idx_0])
hid_probs_1 = ising_probs(h_hidden_1[best_logl_idx_1],
                          J_hidden_1[best_logl_idx_1])
hid_probs_2 = ising_probs(h_hidden_2[best_logl_idx_2],
                          J_hidden_2[best_logl_idx_2])

plot_params(true_probs, hid_probs_neg2, 'x', 0.01) # crazy
plot_params(true_probs, hid_probs_neg1, 'x', 0.01) # crazy
plot_params(true_probs, hid_probs_0, 'x', 0.01) # crazy
plot_params(true_probs, hid_probs_1, 'x', 0.01) # wow; big change
plot_params(true_probs, hid_probs_2, 'x', 0.01) # crazy flat 

# compare p marginalized 
_, true_marginalized = marginalize_n(configurations, true_probs, n_hidden)
_, hid_marginalized_neg2 = marginalize_n(configurations, hid_probs_neg2, n_hidden)
_, hid_marginalized_neg1 = marginalize_n(configurations, hid_probs_neg1, n_hidden)
_, hid_marginalized_0 = marginalize_n(configurations, hid_probs_0, n_hidden)
_, hid_marginalized_1 = marginalize_n(configurations, hid_probs_1, n_hidden)
_, hid_marginalized_2 = marginalize_n(configurations, hid_probs_2, n_hidden)

#plot_params(true_marginalized, vis_marginalized, 'x', 0.01)
plot_params(true_marginalized, hid_marginalized_neg2, 'x', 0.01) # really good
plot_params(true_marginalized, hid_marginalized_neg1, 'x', 0.01) # pretty good
plot_params(true_marginalized, hid_marginalized_0, 'x', 0.01) # pretty good
plot_params(true_marginalized, hid_marginalized_1, 'x', 0.01) # bad
plot_params(true_marginalized, hid_marginalized_2, 'x', 0.01) # terrible 

# deconvert J & check distributions
# we get wild values for all types;
sns.distplot(logl_hidden_neg2)
sns.distplot(logl_hidden_1)

logl_hidden_1

J_hid_neg2_list = []
J_int_neg2_list = []
J_vis_neg2_list = []
for i, _ in enumerate(logl_hidden_neg2): 
    J_hid_neg2, J_int_neg2, J_vis_neg2 = deconstruct_J(J_hidden_neg1[i], 
                                                       n_hidden, 
                                                       n_visible)
    J_hid_neg2_list.append(J_hid_neg2)
    J_int_neg2_list.append(J_int_neg2)
    J_vis_neg2_list.append(J_vis_neg2)
sns.distplot(np.concatenate(J_hid_neg2_list), hist=False, rug=True)
sns.distplot(np.concatenate(J_int_neg2_list), hist=False, rug=True)
sns.distplot(np.concatenate(J_vis_neg2_list), hist=False, rug=True)
np.mean([0.01, 0.25, 0.64])
np.log(0.3) 
###### figure out the J function ########
n = 2
h = np.array([1, 2, 3, 4])
import itertools 
perms = list(itertools.permutations(h[:n]))
combinations = []

# Loop through the permutations and append the constant elements
for perm in perms:
    combination = list(perm) + list(h[n:])
    combinations.append(combination)

# Convert the combinations list to a numpy array
combinations = np.array(combinations)
combinations
