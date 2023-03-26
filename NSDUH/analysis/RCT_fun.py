import numpy as np 
from numba import njit  

@njit
def find_neighbors(input_idx, all_combinations, num_variables, num_flips, enforce, enforce_idx):
    """
    Finds all neighbor indices of the input index that differ by exactly num_flips bit flips.
    NB: only takes up to n=2 bit flips for now 
    """
    n_combinations = len(all_combinations)
    arr_N = np.zeros((n_combinations,), dtype=np.int64)    
    
    if num_flips == 1: 
        arr_E = np.zeros((1,), dtype=np.int64) # 1 flip 
    elif num_flips == 2:
        arr_E = np.zeros((num_variables-1,), dtype=np.int64) # 19 flips 
    
    arr_E_idx = 0
    idx = 0
    for bit_indices in all_combinations:
        bit_mask = 0
        for i in bit_indices:
            bit_mask |= 1 << i
        neighbor = input_idx ^ bit_mask
        arr_N[idx] = neighbor
        if enforce: 
            if num_variables-enforce_idx-1 in bit_indices:
                arr_E[arr_E_idx] = idx
                arr_E_idx += 1
        idx += 1 
    return arr_N, arr_E

@njit
def glauber(p, i, neighbors, num_neighbors, enforce, enforce_idx): 
    '''
    p: array of probabilities
    i: index of current configuration
    neighbors: indices of neighboring configurations
    '''
    p_neighbors = p[neighbors]
    p_self = p[i]
    p_move = p_neighbors / (p_self + p_neighbors)
    p_move = p_move / num_neighbors
    if enforce: 
        p_move[enforce_idx] = 0
    p_stay = 1 - np.sum(p_move)
    return p_move, p_stay

@njit 
def evolve(p_current, p_evolve, idx, p_stay, p_move, neighbors, num_neighbors):
    p_evolve[idx] += p_stay * p_current[idx]
    for n in range(num_neighbors): 
        p_evolve[neighbors[n]] += p_move[n] * p_current[idx]
    return p_evolve 

@njit() # not sure how to parallelize 
def push_forward(num_variables, configurations, 
                 configuration_probabilities, 
                 p_current, all_combinations, 
                 num_flips, num_neighbors, enforce_idx):
    p_evolve = np.zeros_like(p_current)
    if enforce_idx > num_variables: 
        enforce = False
    else: 
        enforce = True 
    for idx in range(len(configurations)): # these could be run in parallel ...
        # find neighbors and flag the ones we enforce
        neighbors, enforce_arr = find_neighbors(idx, all_combinations,
                                                num_variables, num_flips,
                                                enforce, enforce_idx)
        # get probability for each possible move and stay 
        p_move, p_stay = glauber(configuration_probabilities, idx, 
                                 neighbors, num_neighbors, enforce, enforce_arr)
        
        # evolve the probability distribution 
        p_evolve = evolve(p_current, p_evolve, idx, 
                          p_stay, p_move, neighbors,
                          num_neighbors)
    return p_evolve 