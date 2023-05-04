import numpy as np 
import pandas as pd 
import re 
import os 
from sample_functions import read_text_file, plot_params, plot_h_hidden, ising_probs, bin_states
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

# marginalize states 
def marginalize_over_n_elements(configurations, probabilities, n):
    # Remove the first n columns (elements) from configurations
    reduced_configurations = configurations[:, n:]

    # Find the unique configurations in reduced_configurations
    unique_configurations = np.unique(reduced_configurations, axis=0)

    # Initialize an empty list to store the probabilities for each unique configuration
    marginalized_probs = []

    # Loop through unique configurations and sum the probabilities corresponding to the same configuration
    for config in unique_configurations:
        prob = np.sum(probabilities[np.all(reduced_configurations == config, axis=1)])
        marginalized_probs.append(prob)

    # Convert lists to numpy arrays
    marginalized_probs = np.array(marginalized_probs)
    unique_configurations = np.array(unique_configurations)

    return unique_configurations, marginalized_probs

_, true_marginalized = marginalize_over_n_elements(configurations, true_probs, 2)
_, hid_marginalized = marginalize_over_n_elements(configurations, hid_probs, 2)
_, vis_marginalized = marginalize_over_n_elements(configurations, vis_probs, 2)

# still not quite right: 
plot_params(true_marginalized, vis_marginalized, 'x', 0.01)
plot_params(true_marginalized, hid_marginalized, 'x', 0.01)


# figure out the J function
