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

    return samples 


# data in the connected format (h, J)
# annotated to highlight correspondence
# not sure in terms of constructing these from different distributions 

h = np.array([-0.5, # first hidden node 
              0.5, # second hidden node
              0, # first visible node
              0, # second visible node
              0 # third visible node 
              ])

J = np.array([0, # between two hidden 
              -0.5, # first hidden to first non-hidden
              0, # first hidden to second non-hidden
              0.5, # first hidden to third non-hidden  
              0, # second hidden to first non-hidden
              -1, # second hidden to second non-hidden
              0, # second hidden to third non-hidden
              0, # first non-hidden to second non-hidden
              0, # first non-hidden to third non-hidden
              0 # second non-hidden to third non-hidden
              ])

sample = sample_fully_connected(1000, h, J)

np.mean(sample, axis = 0) # the ones we care about are the last three here 
sample_visible = sample[:, 2:]
np.corrcoef(sample_visible, rowvar = False) # looks low to me 

