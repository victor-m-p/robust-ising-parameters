import numpy as np 
from sample_functions import read_text_file, bin_states, ising_probs, marginalize_n, sample_fully_connected
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

# -1.0 
files_neg1 = [x for x in files_hidden if sparsity_regex.search(x).group(0).startswith('-1.0')]
h_hidden_neg1, J_hidden_neg1, logl_hidden_neg1 = load_txt_dir(path_mpf, files_neg1, n_connections)

# best sparsity: 
best_logl_idx_neg1 = np.where(logl_hidden_neg1 == np.max(logl_hidden_neg1))[0][0]
h_hidden_neg1_best = h_hidden_neg1[best_logl_idx_neg1]
J_hidden_neg1_best = J_hidden_neg1[best_logl_idx_neg1]
hJ_hidden_neg1_best = np.concatenate((J_hidden_neg1_best, h_hidden_neg1_best))

# load actual params
param_files = [x for x in os.listdir(path_true) if x.startswith('format')]
filename = param_files[0]
hJ_true = np.loadtxt(f'{path_true}{filename}')

def find_indices(A: np.array, 
                 B: np.array): 
    '''
    A: all configurations
    B: observed configurations
    '''
    # Find the indices of the first matches in A for each row in B
    matches = np.all(A[:, None] == B, axis=-1)
    indices = np.argmax(matches, axis=0)
    return indices

# calculate log likelihood of each
def logl(params, data, n_nodes, n_hidden = 0):
    # take out params 
    nj = int(n_nodes*(n_nodes-1)/2)
    h = params[nj:]
    J = params[:nj]

    # all configurations
    configs = bin_states(n_nodes)

    # calculate probabilities
    true_probs = ising_probs(h, J)

    # calculate marginalized probabilities
    configs_marginal, probs_marginal = marginalize_n(configs, true_probs, n_hidden)

    # take out first n rows of data 
    data_marginal = data[:, n_hidden:]

    # calculate log likelihood
    indices = find_indices(configs_marginal, data_marginal)
    probabilities = probs_marginal[indices]
    logprobs = np.log(probabilities)
    sumlogprobs = np.sum(logprobs)
    
    return sumlogprobs

# logl of best fit 
logl_hidden_marginalized = logl(hJ_hidden_neg1_best, 
                                data, 
                                n_nodes, 
                                n_hidden)

logl_hidden_marginalized # -26808.42 --- hmmmmm

# logl of true params
logl_true_marginalized = logl(hJ_true,
                              data,
                              n_nodes,
                              n_hidden) 

logl_true_marginalized # -26832.70 --- hmmmm 

# but then with the penalty term: 
def penalty(params, sparsity, norm): 
    return (83)*(10**sparsity)*np.sum(np.abs(params**norm))

penalty_hidden_marginalized = penalty(hJ_hidden_neg1_best, -1.0, 2) # 557
penalty_true_marginalized = penalty(hJ_true, -1.0, 2) # 5.57

loss_hidden_marginalized = logl_hidden_marginalized - penalty_hidden_marginalized # -27365.77
loss_true_marginalized = logl_true_marginalized - penalty_true_marginalized # -26838.28

# difference in probability
np.abs(loss_hidden_marginalized - loss_true_marginalized) 

# what could be going wrong?
# (1) marginal 
# (2) my calculations 
# (3) wrong calculation the true parameters (but that would not affect the logl for inferred). 
# (4) wrong calculation for p I guess. 

### the below is great; need to save somewhere more permanent. 
# simulate data from the inferred parameters 
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