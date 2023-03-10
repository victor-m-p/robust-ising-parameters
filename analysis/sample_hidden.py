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
n = 10
C = 500
scale = 1.0
np.random.seed(1)
invlogit = lambda x: 1 / (1 + np.exp(-x))

# generate params and probabilities
h = np.random.normal(scale=scale, size=n)
J = np.random.normal(scale=scale, size=n*(n-1)//2)
hJ = np.concatenate((h, J))
probabilities = p_dist(h, J) # potential culprit 

# save probabilities 
Jh = np.concatenate((J, h))
np.savetxt(f"../data/hidden_nodes/questions_{n}_samples_{C}_scale_{scale}_true_100.dat", Jh)

# get a sample 
allstates = bin_states(n, True)  # all 2^n possible binary states in {-1,1} basis
sample = allstates[np.random.choice(range(2**n), # doesn't have to be a range
                                    size=C, # how many samples
                                    replace=True, # a value can be selected multiple times
                                    p=probabilities)]  # random sample from p(s)

# save complete data 
bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in sample]
weight_string = [str(1.0) for _ in range(C)]
save_dat(bit_string, weight_string, sample,
         f'../data/hidden_nodes/questions_{n}_samples_{C}_scale_{scale}_complete_100.dat')

# save data with one additional node 
sample_extra = np.insert(sample, n, 0, axis=1)
bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in sample_extra]
save_dat(bit_string, weight_string, sample_extra,
         f'../data/hidden_nodes/questions_{n}_samples_{C}_scale_{scale}_extra_100.dat')

# save all n-1 (add hidden node)
_, columns = sample.shape
for i in range(columns): 
    sample_i = np.copy(sample)
    sample_i[:, i] = 0 
    bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in sample_i]
    save_dat(bit_string, weight_string, sample_i,
             f'../data/hidden_nodes/questions_{n}_samples_{C}_scale_{scale}_hidden_{i}.dat')

# two hidden nodes
for i in range(columns-1): 
    sample_i = np.copy(sample)
    sample_i[:, i] = 0
    sample_i[:, i+1] = 0
    bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in sample_i]
    save_dat(bit_string, weight_string, sample_i,
             f'../data/hidden_nodes/questions_{n}_samples_{C}_scale_{scale}_twohidden_{i}.dat')

# save all n-1 (remove node)
for i in range(columns): 
    sample_i = np.delete(sample, i, axis=1)
    bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in sample_i]
    save_dat(bit_string, weight_string, sample_i,
             f"../data/hidden_nodes/questions_{n}_samples_{C}_scale_{scale}_removed_{i}.dat")

# delete two nodes  
for i in range(columns-1): 
    sample_i = np.delete(sample, np.s_[i:i+2], axis = 1)
    bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in sample_i]
    save_dat(bit_string, weight_string, sample_i,
            f"../data/hidden_nodes/questions_{n}_samples_{C}_scale_{scale}_tworemoved_{i}.dat")