'''
VMP 2023-04-28: 
The simple case but with couplings in the hidden layer 
(1) only intralayer connections
(2) all layers are binary
(3) hidden features not independent 
'''

import numpy as np 
import itertools 

# taken from coniii enumerate
def fast_logsumexp(X, coeffs=None):
    """correlation calculation in Ising equation

    Args:
        X (np.Array): terms inside logs
        coeffs (np.Array, optional): factors in front of exponential. Defaults to None.

    Returns:
        float: sum of exponentials
    """
    Xmx = max(X)
    if coeffs is None:
        y = np.exp(X-Xmx).sum()
    else:
        y = np.exp(X-Xmx).dot(coeffs)

    if y<0:
        return np.log(np.abs(y))+Xmx, -1.
    return np.log(y)+Xmx, 1.

# still create J_combinations is slow for large number of nodes
def p_dist(h, J):
    """return probabilities for 2**h states

    Args:
        h (np.Array): local fields
        J (np.Array): pairwise couplings. 

    Returns:
        np.Array: probabilities for all configurations
    """
    n_nodes = len(h)
    hJ = np.concatenate((h, J))
    h_combinations = np.array(list(itertools.product([1, -1], repeat = n_nodes)))
    J_combinations = np.array([list(itertools.combinations(i, 2)) for i in h_combinations])
    J_combinations = np.add.reduce(J_combinations, 2)
    J_combinations[J_combinations != 0] = 1
    J_combinations[J_combinations == 0] = -1
    condition_arr = np.concatenate((h_combinations, J_combinations), axis = 1)
    flipped_arr = hJ * condition_arr
    summed_arr = np.sum(flipped_arr, axis = 1)
    logsumexp_arr = fast_logsumexp(summed_arr)[0]
    Pout = np.exp(summed_arr - logsumexp_arr)
    return Pout[::-1] # should it be reversed? (yes). 

def bin_states(n):
    """generate 2**n possible configurations

    Args:
        n (int): number of questions (features)

    Returns:
        np.Array: 2**n configurations 
    """
    v = np.array([list(np.binary_repr(i, width=n)) for i in range(2**n)]).astype(int)
    return v*2-1

# 1. the simplest case: 
n_sample = 10000 # all independent (no repeated measurement)
n_visible = 3 # number of visible units
n_hidden = 2 # number of hidden units 
weight_distribution = np.random.normal(0, 5, (n_visible, n_hidden)) # weight from hidden to visible layer

# this is basically our bias distribution 
h = np.random.normal(loc=0, scale=1, size=n_hidden)
J = np.random.normal(loc=0, scale=1, size=n_hidden*(n_hidden-1)//2)
probabilities = p_dist(h, J) 
allstates = bin_states(n_hidden, True) # this needs to be 1/0

# run 
samp_visible = np.zeros((n_sample, n_visible)) # init samples of visible units
for i in range(n_sample): 
    error_sample = np.random.normal(0, 1, (n_visible, n_hidden)) # error term
    weight_sample = weight_distribution + error_sample # error on weight 
    bias_sample = allstates[np.random.choice(a=np.arange(4), p=probabilities)] # bias of hidden layer
    samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(bias_sample, weight_sample.T))))) # yi

np.mean(samp_visible, axis=0) 
np.corrcoef(samp_visible, rowvar=False) 

# try to code it up 
def intralayer_dependent_binary(n_sample,
                                n_visible,
                                n_hidden,
                                h,
                                J,
                                weight_params = (0, 1),
                                error_params = (0, 1)):
    # distribution params 
    e_mean, e_sd = error_params
    w_mean, w_sd = weight_params
    weight_distribution = np.random.normal(w_mean, w_sd, (n_visible, n_hidden)) 
    # Glauber
    probabilities = p_dist(h, J) 
    allstates = bin_states(n_hidden) 
    n_states = len(allstates)
    samp_visible = np.zeros((n_sample, n_visible))
    for i in range(n_sample): 
        error_sample = np.random.normal(e_mean, e_sd, (n_visible, n_hidden))
        weight_sample = weight_distribution + error_sample 
        bias_sample = allstates[np.random.choice(a=np.arange(n_states), p=probabilities)]
        samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(bias_sample, weight_sample.T)))))
    return samp_visible

# can we obtain a correlated system?
n_sample = 1000 # all independent (no repeated measurement)
n_visible = 5 # number of visible units
n_hidden = 3 # number of hidden units 
weight_distribution = np.random.normal(0, 1, (n_visible, n_hidden)) # weight from hidden to visible layer
weight_distribution.shape

# this is basically our bias distribution 
h = np.random.normal(loc=0, scale=1, size=n_hidden)
J = np.random.normal(loc=0, scale=1, size=n_hidden*(n_hidden-1)//2)
probabilities = p_dist(h, J) 
allstates = bin_states(n_hidden, True)

# check correlation
sample = intralayer_dependent_binary(n_sample, n_visible, n_hidden, h, J)
np.corrcoef(sample, rowvar=False)

# again we basically do not get correlation

# only need full calculation in some cases. 
# we should formulate this into simply functions. 

########## starts here ###########
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

# make into one nice function
def sample_independent(n_samples: int,
                       n_visible: int,
                       h_hidden: np.array,
                       h_visible: np.array,
                       J_inter: np.array):
    
    # probability of hidden nodes being on/off
    prob_hidden = np.array(list(map(expnorm, h_hidden)))
    
    # sample array 
    samples = np.zeros((n_samples, n_visible))
    
    # sample hidden & recode to 1/-1
    for i in range(n_samples): 
        samp_hidden = np.random.binomial(1, prob_hidden)
        samp_hidden = recode(samp_hidden)

        # sample visible & recode to 1/-1
        samp_h_visible = h_visible + np.sum(samp_hidden * J_inter, axis=1)
        samp_visible = np.random.binomial(1, expnorm(samp_h_visible))
        samp_visible = recode(samp_visible)
        
        samples[i] = samp_visible
    return samples 

n_hidden = 2
n_visible = 3 
J_hidden = 0 # independent hidden layer
h_hidden = np.array([-0.5, 0.5])
J_visible = 0 # independent visible layer  
h_visible = np.array([0, 0, 0]) 
J_between = np.array([[1, 0], [0, -1], [0.5, 0]])

samp = sample_independent(1000, 
                          n_hidden, 
                          n_visible, 
                          h_hidden, 
                          h_visible, 
                          J_between)

# looks good I think 
# question is how this compares to other models 
np.mean(samp, axis=0)
np.corrcoef(samp, rowvar=False) # still only .25 correlations 

# notes: 
# do I have to sample the hidden layer?
# seems like I should be able to just 
# directly calculate the probabilities (once)
# for the visible layer and just sample from that 
# but perhaps that does not quite work. 