'''
Helper functions for the analysis of DRH data (and preprocessing)
VMP 2022-02-06: refactored with chatGPT and docstrings. 
'''

import numpy as np
import itertools 
import re 
import matplotlib.pyplot as plt 

# basic simulation functions 
def ising_probs(h: np.array, 
                J: np.array):
    """return probabilities for 2**h states

    Args:
        h (np.Array): local fields
        J (np.Array): pairwise couplings. 

    Returns:
        np.Array: probabilities for all configurations
    """
    
    # all h combinations
    h_combinations = np.array(list(itertools.product([1, -1], repeat = len(h))))
    
    # all J combinations
    J_combinations = np.array([list(itertools.combinations(i, 2)) for i in h_combinations])
    J_combinations = np.add.reduce(J_combinations, 2)
    J_combinations[J_combinations != 0] = 1
    J_combinations[J_combinations == 0] = -1
    
    # create condition array 
    condition_array = np.concatenate((J_combinations, h_combinations), axis = 1)
    
    # run the calculations
    Jh = np.concatenate((J, h))
    inner_sums = Jh * condition_array 
    total_sums = np.sum(inner_sums, axis = 1)
    exponent_sums = np.exp(total_sums)
    Z = np.sum(exponent_sums)
    p = exponent_sums / Z
    return p[::-1] # reverse because states ascending  

def bin_states(n: int):
    """generate 2**n possible configurations
    
    Args:
        n (int): number of questions (features)
    Returns:
    
        np.Array: 2**n configurations 
    """
    v = np.array([list(np.binary_repr(i, width=n)) for i in range(2**n)]).astype(int)
    return v*2-1

def recode(x): 
    '''
    recode from 0/1 to 1/-1
    '''
    return x*2-1

def expnorm(x):
    '''
    from h to probability 
    '''
    return np.exp(x) / (np.exp(x) + np.exp(-x))

# sample functions 
def sample_not_connected(n_samples: int,
                         h_hidden: np.array,
                         h_visible: np.array,
                         J_interlayer: np.array):
    
    # calculate basic stuff 
    n_visible = len(h_visible)
    n_hidden = len(h_hidden)
    
    # probability of hidden nodes being on/off
    prob_hidden = np.array(list(map(expnorm, h_hidden)))
    
    # sample array 
    samples = np.zeros((n_samples, n_hidden + n_visible))
    
    # sample hidden & recode to 1/-1
    for i in range(n_samples): 
        samp_hidden = np.random.binomial(1, prob_hidden)
        samp_hidden = recode(samp_hidden)

        # sample visible & recode to 1/-1
        samp_h_visible = h_visible + np.sum(samp_hidden * J_interlayer, axis=1)
        samp_visible = np.random.binomial(1, expnorm(samp_h_visible))
        samp_visible = recode(samp_visible)
        
        samples[i] = np.concatenate((samp_hidden, samp_visible))
    return samples.astype(int)

def sample_hidden_connected(n_samples: int,
                            h_hidden: np.array,
                            J_hidden: np.array,
                            h_visible: np.array,
                            J_interlayer: np.array):
    
    # calculate basic stuff 
    n_visible = len(h_visible)
    n_hidden = len(h_hidden)
    
    # probability of hidden nodes being on/off 
    p_hidden = ising_probs(h_hidden, J_hidden)
    
    # all possible states for hidden nodes
    hidden_states = bin_states(n_hidden)
    n_hidden_states = len(hidden_states)
    
    # sample array 
    samples = np.zeros((n_samples, n_hidden + n_visible))

    # sample hidden & recode to 1/-1
    for i in range(n_samples): 
        samp_hidden = hidden_states[np.random.choice(a=np.arange(n_hidden_states), p=p_hidden)]        

        # sample visible & recode to 1/-1
        samp_h_visible = h_visible + np.sum(samp_hidden * J_interlayer, axis=1)
        samp_visible = np.random.binomial(1, expnorm(samp_h_visible))
        samp_visible = recode(samp_visible)

        # concatenate
        samples[i] = np.concatenate((samp_hidden, samp_visible))
        
    return samples.astype(int) 

def sample_fully_connected(n_samples: int,
                           h: np.array,
                           J: np.array):
    
    # calculate basic stuff
    n_nodes = len(h)
    
    # probability of hidden nodes being on/off 
    p = ising_probs(h, J)
    
    # all possible states for hidden nodes
    states = bin_states(n_nodes)
    n_states = len(states)
    
    # sample array 
    samples = np.zeros((n_samples, n_nodes))

    # sample hidden & recode to 1/-1
    for i in range(n_samples): 
        samples[i] = states[np.random.choice(a=np.arange(n_states), p=p)]

    return samples.astype(int)

# saving samples to mpf readable format
def save_to_mpf(simulation_matrix, conversion_dict, outname, weight_list = []): 

    # convert to bitstring in mpf format
    bitstring_list = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in simulation_matrix]

    # extract shape
    rows, cols = simulation_matrix.shape
    
    # if weight_list is empty then fill with ones 
    if not weight_list: 
        weight_list = np.ones(rows)
    
    with open(f'{outname}', 'w') as f: 
        f.write(f'{rows}\n{cols}\n')
        for bit, weight in zip(bitstring_list, weight_list): 
            f.write(f'{bit} {weight}\n')

# Read the text file
def read_text_file(filename, 
                   params_pattern = r'params=\[([\d\s.,e-]+)\]', 
                   logl_pattern = r'Total LogL for data, given parameters: ([-\d.]+)'):
    with open(filename, 'r') as f:
        for line in f:
            # Check for params line
            params_match = re.match(params_pattern, line.strip())
            if params_match:
                values = params_match.group(1)
                params = np.array([float(value) for value in re.split(',\s*', values)])
            
            # Check for LogL line
            logl_match = re.match(logl_pattern, line.strip())
            if logl_match:
                logl = float(logl_match.group(1))
            
        return params, logl


# plotting function
def plot_params(params_true,
                params_inf,
                title,
                constant):
    min_lim = np.min(np.concatenate((params_true, params_inf))) - constant
    max_lim = np.max(np.concatenate((params_true, params_inf))) + constant
    plt.scatter(params_true, params_inf)
    plt.plot([min_lim, max_lim], [min_lim, max_lim], 'k--')
    plt.xlabel('true')
    plt.ylabel('inferred')
    plt.suptitle(title)

def plot_h_hidden(params_true,
                  params_inf,
                  n_hidden,
                  title,
                  constant):
    min_lim = np.min(np.concatenate((params_true, params_inf))) - constant
    max_lim = np.max(np.concatenate((params_true, params_inf))) + constant
    params_true_hidden = params_true[:n_hidden]
    params_inf_hidden = params_inf[:n_hidden]
    params_true_visible = params_true[n_hidden:]
    params_inf_visible = params_inf[n_hidden:]
    plt.scatter(params_true_visible, params_inf_visible, color='tab:blue', label='visible')
    plt.scatter(params_true_hidden, params_inf_hidden, color='tab:orange', label='hidden')
    plt.plot([min_lim, max_lim], [min_lim, max_lim], 'k--')
    plt.xlabel('true')
    plt.ylabel('inferred')
    plt.suptitle(title)

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

