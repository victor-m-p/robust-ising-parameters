import numpy as np 
import itertools 

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

# we have already validated so lets move on to implementation 
def sample_mixed(n_samples: int,
                 h_hidden: np.array,
                 J_hidden: np.array,
                 h_visible: np.array,
                 J_inter: np.array):
    
    # calculate basic stuff 
    n_visible = len(h_visible)
    n_hidden = len(h_hidden)
    
    # probability of hidden nodes being on/off 
    p_hidden = ising_probs(h_hidden, J_hidden)
    
    # all possible states for hidden nodes
    hidden_states = bin_states(n_hidden)
    n_hidden_states = len(hidden_states)
    
    # sample array 
    samples = np.zeros((n_samples, n_visible))

    # sample hidden & recode to 1/-1
    for i in range(n_samples): 
        samp_hidden = hidden_states[np.random.choice(a=np.arange(n_hidden_states), p=p_hidden)]
        #samp_hidden = recode(samp_hidden) # are we sure that we recode this?

        # sample visible & recode to 1/-1
        samp_h_visible = h_visible + np.sum(samp_hidden * J_inter, axis=1)
        samp_visible = np.random.binomial(1, expnorm(samp_h_visible))
        samp_visible = recode(samp_visible)
        
        samples[i] = samp_visible
    return samples 

n_hidden = 2
n_visible = 3 
J_hidden = np.array([0]) # independent hidden layer
h_hidden = np.array([-0.5, 0.5])
J_visible = 0 # independent visible layer  
h_visible = np.array([0, 0, 0]) 
J_interlayer = np.array([[1, 0], [0, -1], [0.5, 0]])

sample_mixed(10, 
             n_visible, 
             h_hidden, 
             J_hidden, 
             h_visible, 
             J_interlayer)
