import numpy as np 
from fun import p_dist, bin_states, fast_logsumexp
from sample_fun import save_dat, randomword
import pandas as pd 

# convert to bit string
conversion_dict = {
    '-1': '0',
    '0': 'X',
    '1': '1'
}

# N = 10 system
n_nodes = 10
n_samples = 500
scale = 0.5
rw = randomword(10)
#np.random.seed(1)
invlogit = lambda x: 1 / (1 + np.exp(-x))

# generate params and probabilities
h = np.random.normal(scale=scale, size=n_nodes)
J = np.random.normal(scale=scale, size=n_nodes*(n_nodes-1)//2)
hJ = np.concatenate((h, J))
probabilities = p_dist(h, J) # potential culprit 

# save probabilities 
Jh = np.concatenate((J, h))
np.savetxt(f"../data/sim/bias_true/questions_{n_nodes}_samples_{n_samples}_scale_{scale}_true_{rw}.dat", Jh)

# get a sample 
allstates = bin_states(n_nodes, True)  # all 2^n possible binary states in {-1,1} basis
sample = allstates[np.random.choice(range(2**n_nodes), # doesn't have to be a range
                                    size=n_samples, # how many samples
                                    replace=True, # a value can be selected multiple times
                                    p=probabilities)]  # random sample from p(s)

# save complete data 
bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in sample]
weight_string = [str(1.0) for _ in range(n_samples)]
save_dat(bit_string, weight_string, sample,
         f'../data/sim/bias_mpf/questions_{n_nodes}_samples_{n_samples}_scale_{scale}_complete_{rw}.dat')

# remove some combinations
def filter_array(arr, filter_dict, pct=0):
    # create an array of all True values
    bool_array = np.full(shape=arr.shape[0], fill_value=True, dtype=bool)
    
    # loop through the filter_dict and create boolean arrays for each position
    for pos, val in filter_dict.items():
        bool_array = bool_array & (arr[:, pos] == val)
    
    if pct>0: 
        true_indices = np.where(bool_array==True)[0]
        num_to_change = int(len(true_indices) * pct)
        indices_to_change = np.random.choice(true_indices, size=num_to_change, replace=False)
        bool_array[indices_to_change] = False 
        
    # use the boolean array to filter the original array
    filtered_array = arr[~bool_array]
    
    return filtered_array #filtered_array 

filter_dict = {0: 1, 1: -1}
for i in [0, 0.1, 0.5]: 
    filter_sample = filter_array(sample, filter_dict, pct=i)
    n_samples, columns = filter_sample.shape
    bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in filter_sample]
    weight_string = [str(1.0) for _ in range(n_samples)]
    save_dat(bit_string, weight_string, filter_sample,
            f'../data/sim/bias_mpf/questions_{n_nodes}_samples_{n_samples}_scale_{scale}_filtered_{rw}.dat')