import numpy as np 
from sample_functions import read_text_file, bin_states, ising_probs, marginalize_n, sample_fully_connected, find_indices, logl, DKL
import os 
import re 

# sanity check 
f = 'data/fully_connected_true_big/sim_true_nhid_0_nvis_11_th_gaussian_0.0_0.1_tj_gaussian_0.0_0.1_nsim_5000.txt'
data = np.loadtxt(f, dtype='int')

# meta setup
n_nodes = 11
n_connections = int(n_nodes*(n_nodes-1)/2)
n_hidden = 3
n_visible = 8

# load inferred params
figpath = 'fig/fully_connected/'
path_mpf = 'data/fully_connected_big_grid/'
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

# compare samples 
hidden_samples = sample_fully_connected(n_samples=5000,
                                        h=h_hidden_neg1_best,
                                        J=J_hidden_neg1_best)
hidden_samples = hidden_samples[:, n_hidden:]

newtrue_samples = sample_fully_connected(n_samples=5000,
                                         h=hJ_true[n_connections:],
                                         J=hJ_true[:n_connections])
newtrue_samples = newtrue_samples[:, n_hidden:]

data_marginal = data[:, n_hidden:]

# count up number for each unique configuration
configs_marginal = bin_states(n_visible)

def check_samples(samples, possibilities):

    # Count occurrences
    occurrences = np.sum(np.all(samples[:, None] == possibilities, axis=-1), axis=0)

    # Create the observed dictionary
    observed = {i: count for i, count in enumerate(occurrences)}

    # Calculate the normalized counts
    total_samples = samples.shape[0]
    normalized_counts = occurrences / total_samples

    # Create the normalized dictionary
    normalized = {i: count for i, count in enumerate(normalized_counts)}

    return observed, normalized 

observed_true, normalized_true = check_samples(data_marginal, configs_marginal)
observed_newtrue, normalized_newtrue = check_samples(newtrue_samples, configs_marginal)
observed_hidden, normalized_hidden = check_samples(hidden_samples, configs_marginal)

# compare the two:
import matplotlib.pyplot as plt 
fig, ax = plt.subplots()
plt.scatter(normalized_true.values(), 
            normalized_hidden.values(), 
            color='tab:blue',
            alpha=0.5)
plt.scatter(normalized_true.values(), 
            normalized_newtrue.values(), 
            color='tab:orange',
            alpha=0.5)
plt.plot([0, 0.02], [0, 0.02], 'k--')
plt.show(); 

def compare_loss(x, y): 
    ME = np.mean([np.abs(a-b) for a, b in zip(list(x.values()), list(y.values()))])
    MSE = np.mean([(a-b)**2 for a, b in zip(list(x.values()), list(y.values()))])
    return ME, MSE 

me_hidden, mse_hidden = compare_loss(normalized_true, normalized_hidden) 
me_newt, mse_newt = compare_loss(normalized_true, normalized_newtrue)