import numpy as np 
import pandas as pd 
import re 
import os 
from sample_functions import read_text_file, plot_params, plot_h_hidden, ising_probs, bin_states, marginalize_over_n_elements
import matplotlib.pyplot as plt 

# meta setup
n_nodes = 6
n_connections = int(n_nodes*(n_nodes-1)/2)

# match the files
figpath = 'fig/fully_connected/'
path_mpf = 'data/fully_connected_mpf/'
path_true = 'data/fully_connected_true/'

# load files helper  
def load_txt_dir(path, files):
    h_list = []
    J_list = []
    logl_list = []
    for filename in files: 
        params, logl = read_text_file(f"{path}{filename}")
        logl_list.append(logl)
        h_list.append(params[n_connections:])
        J_list.append(params[:n_connections])
    return h_list, J_list, logl_list

files_hidden = [x for x in os.listdir(path_mpf) if x.endswith('_log.txt') and x.startswith('sim_hid')]
h_hidden, J_hidden, logl_hidden = load_txt_dir(path_mpf, files_hidden)

files_visible = [x for x in os.listdir(path_mpf) if x.endswith('_log.txt') and x.startswith('sim_vis')]
h_visible, J_visible, logl_visible = load_txt_dir(path_mpf, files_visible)

# read actual shit 
param_files = [x for x in os.listdir(path_true) if x.startswith('format')]
filename = param_files[0]
params_true = np.loadtxt(f'{path_true}{filename}')
h_true = params_true[n_connections:]
J_true = params_true[:n_connections]

# compare h (hmmm)
n_hidden = 2
ele = 0
plot_h_hidden(h_true, h_hidden[ele], n_hidden, 'x', 0.1)
plot_h_hidden(h_true, h_visible[ele], n_hidden, 'x', 0.1) # pretty good 

# compare J 
plot_params(J_true, J_visible[ele], 'x', 0.1) # what?
plot_params(J_true, J_hidden[ele], 'x', 0.1) # we do not even need to look into this. 

# compare p states 
ele = 0 
configurations = bin_states(n_nodes)
true_probs = ising_probs(h_true, J_true)
hid_probs = ising_probs(h_hidden[ele], J_hidden[ele])
vis_probs = ising_probs(h_visible[ele], J_visible[ele])

plot_params(true_probs, vis_probs, 'x', 0.01)
plot_params(true_probs, hid_probs, 'x', 0.01)

_, true_marginalized = marginalize_over_n_elements(configurations, true_probs, 2)
_, hid_marginalized = marginalize_over_n_elements(configurations, hid_probs, 2)
_, vis_marginalized = marginalize_over_n_elements(configurations, vis_probs, 2)

plot_params(true_marginalized, vis_marginalized, 'x', 0.01)
plot_params(true_marginalized, hid_marginalized, 'x', 0.01)




# figure out the J function
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

# how about the J?


def best_hidden_fit(p_true, p_hidden): 
