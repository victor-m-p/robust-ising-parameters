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

def deconstruct_J(J, n_hidden, n_visible):
    '''
    deconstruct J into J_hidden, J_inter, J_visible.
    needed to compare with inferred J from mpf.
    '''
    J_hidden = []
    J_inter = []
    idx = 0
    n = 1
    for i in range(n_hidden): 
        J_hidden += list(J[idx:idx+n_hidden-n])
        idx += n_hidden-n
        J_inter += list(J[idx:idx+n_visible])
        idx += n_visible 
        n += 1
    J_visible = J[idx:]
    return np.array(J_hidden), np.array(J_inter), J_visible 

def construct_J(J_hidden, J_inter, J_visible, n_hidden, n_visible):
    '''
    construct J from J_hidden, J_inter, J_visible
    needed to compare with inferred J from mpf. 
    '''
    idx_hidden = 0
    idx_inter = 0
    J_list = []
    n = 1
    for i in range(n_hidden):
        J_list += list(J_hidden[idx_hidden:idx_hidden+n_hidden-n])
        J_list += list(J_inter[idx_inter:idx_inter+n_visible])
        idx_hidden+=n_hidden-n
        idx_inter+=n_visible 
        n += 1

    J_list += list(J_visible)
    return J_list


# sample functions 
def sample_not_connected(n_samples: int,
                         h_hidden: np.array,
                         h_visible: np.array,
                         J_interlayer: np.array):
    '''
    sample Ising data from a system that is independent in both layers.
    n_samples: number of samples desired.
    h_hidden: local fields for hidden nodes.
    h_visible: local fields for visible nodes.
    J_interlayer: pairwise couplings between hidden and visible nodes.
    '''
    
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
    '''
    sample Ising data from a system where there are connections
    between the hidden nodes and between the hidden nodes and the visible nodes
    but not between the visible nodes. 
    n_samples: number of samples desired.
    h_hidden: local fields for hidden nodes.
    J_hidden: pairwise couplings between hidden nodes.
    h_visible: local fields for visible nodes.
    J_interlayer: pairwise couplings between hidden and visible nodes.
    '''
    
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
    '''
    sample Ising data from a fully connected system.
    n_samples: number of samples desired.
    h: local fields
    J: pairwise couplings.
    '''
    
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
def save_to_mpf(simulation_matrix: np.array, 
                conversion_dict: dict, 
                outname: str, 
                weight_list = []): 
    '''
    convert a numpy array of 1/0/-1 to mpf format (conversion dict) and save. 
    optionally include a list of weights. 
    '''
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
def read_text_file(filename: str, 
                   params_pattern = r'params=\[([\d\s.,e+-]+)\]', 
                   logl_pattern = r'Total LogL for data, given parameters: ([-\d.]+)'):
    '''
    read a text file output from the 'mpf -l' option.
    get the parameters and logl and return. 
    '''
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
def plot_params(params_true: np.array,
                params_inf: np.array,
                title: str,
                constant = 0.1):
    '''
    plot the true parameters against the inferred parameters.
    give the plot a title & constant for the axis limits.
    '''
    min_lim = np.min(np.concatenate((params_true, params_inf))) - constant
    max_lim = np.max(np.concatenate((params_true, params_inf))) + constant
    plt.scatter(params_true, params_inf)
    plt.plot([min_lim, max_lim], [min_lim, max_lim], 'k--')
    plt.xlabel('true')
    plt.ylabel('inferred')
    plt.suptitle(title)

def plot_h_hidden(params_true: np.array,
                  params_inf: np.array,
                  n_hidden: int,
                  title: str,
                  constant = 0.1):
    '''
    plot the true (h) parameters against the inferred parameters (with hidden states).
    give the plot a title & constant for the axis limits.
    '''
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

def marginalize_n(configurations: np.array, probabilities: np.array, n_hidden: int):
    '''
    marginalize the hidden states and return the marginalized 
    unique configurations and probabilities.
    it is assumed that the hidden nodes are the first n columns.
    '''
    # Remove the first n columns (elements) from configurations
    reduced_configurations = configurations[:, n_hidden:]

    # Find the unique configurations in reduced_configurations and their inverse
    unique_configurations, inverse = np.unique(reduced_configurations, axis=0, return_inverse=True)

    # Sum the probabilities corresponding to the same configuration using bincount
    marginalized_probs = np.bincount(inverse, weights=probabilities)

    return unique_configurations, marginalized_probs


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

def match_idx_to_prob(question_idx: int,
                      states: np.array,
                      p: np.array):
    '''
    question_idx: column index of the question 
    states: all possible states
    p: probabilities for all possible states
    Returns the probability of the question being 0 or 1 for each state.
    '''

    n_states = states.shape[0]
    n_splits = 2**(question_idx+1)
    n_states_split = int(n_states/n_splits)

    p_neg = []
    p_pos = []
    n_states_running = 0
    while n_states_running < n_states: 

        idx_neg = np.array(range(n_states_running, 
                                n_states_split+n_states_running))

        idx_pos = np.array(range(n_states_running+n_states_split, 
                                n_states_running+n_states_split*2))

        p_neg += list(p[idx_neg])
        p_pos += list(p[idx_pos])

        n_states_running += n_states_split*2

    matched_p = np.column_stack((p_neg, p_pos))
    return matched_p 

def param_magnitude_sum(params: np.ndarray,
                        norm: float = 1.0):
    '''
    params: inferred or true parameters. 
    returns the magnitude of the parameters. 
    '''
    return np.sum(np.abs(params)**norm)

def param_magnitude_mean(params: np.ndarray,
                         norm: float = 1.0):
    '''
    params: inferred or true parameters. 
    returns the magnitude of the parameters. 
    '''
    return np.mean(np.abs(params)**norm)

def regularization_penalty(params: np.ndarray, 
                           sparsity: float, 
                           norm: float): 
    '''
    params: parameters inferred or true. 
    sparsity: lambda. 
    norm: 1 (L1) or 2 (L2) norm (or any other q-norm). 
    returns the additional penalty by the regularization. 
    '''
    return (83)*(10**sparsity)*np.sum(np.abs(params**norm))

def logl(params: np.ndarray, 
         data: np.ndarray, 
         n_nodes: int, 
         n_hidden = 0,
         configs = None):
    
    '''
    calculates the log likelihood of data given parameters.
    can marginalize in the case of hidden nodes. 
    '''

    # take out params 
    nj = int(n_nodes*(n_nodes-1)/2)
    h = params[nj:]
    J = params[:nj]

    # all configurations
    if configs is None:
        configs = bin_states(n_nodes) 

    # calculate probabilities
    true_probs = ising_probs(h, J)
    
    # take out marginal data
    data_marginal = data[:, n_hidden:]

    # calculate marginalized probabilities
    if n_hidden > 1: 
        configs_marginal, probs_marginal = marginalize_n(configs, true_probs, n_hidden)
    else: 
        configs_marginal = configs 
        probs_marginal = true_probs
        
    # calculate log likelihood
    indices = find_indices(configs_marginal, data_marginal) # overlap in indices 
    probabilities = probs_marginal[indices] # probability for these indices 
    sumlogprobs = np.sum(np.log(probabilities))
    
    return sumlogprobs

def DKL(params_true: np.ndarray, 
        params_model: np.ndarray, 
        n_nodes: int,
        n_hidden = 0): 
    '''
    params_true: true parameters (when known)
    params_model: inferred parameters
    n_nodes: number of nodes in the system
    n_hidden: number of hidden nodes (assumes they are the first columns).
    returns: D_KL(P_true||P_model) which can be thought of as the excess surprise
    from using P_model when the actual distribution is P_true.  
    '''

    # take out params 
    nj = int(n_nodes*(n_nodes-1)/2)
    h_true = params_true[nj:]
    J_true = params_true[:nj]
    h_model = params_model[nj:]
    J_model = params_model[:nj]

    # all configurations
    configs = bin_states(n_nodes)

    # calculate probabilities
    true_probs = ising_probs(h_true, J_true)
    model_probs = ising_probs(h_model, J_model)

    # marginalize 
    _, true_probs_marginal = marginalize_n(configs, true_probs, n_hidden)
    _, model_probs_marginal = marginalize_n(configs, model_probs, n_hidden)

    # calculate KL divergence
    kl = np.sum(true_probs_marginal*np.log(true_probs_marginal/model_probs_marginal))
    return kl 

