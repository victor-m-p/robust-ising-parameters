import numpy as np 
from fun import p_dist, bin_states, fast_logsumexp
from sample_fun import save_dat, randomword
import pandas as pd 

## test
d = pd.read_csv('/home/vmp/robust-ising-parameters/data/reference/direct_reference_questions_20_maxna_5_nrows_455_entries_407.csv')
d = d.drop(columns = ['entry_id', 'weight'])
A = d.to_numpy()
len(np.where(~A.all(axis=1))[0])

conversion_dict = {
    '-1': '0',
    '0': 'X',
    '1': '1'
}

# N = 10 system
n = 10
C = 500
scale = 0.5
np.random.seed(0)
invlogit = lambda x: 1 / (1 + np.exp(-x))

# generate params and probabilities
h = np.random.normal(scale=scale, size=n)
J = np.random.normal(scale=scale, size=n*(n-1)//2)
hJ = np.concatenate((h, J))
probabilities = p_dist(h, J) # potential culprit 

# save probabilities 
Jh = np.concatenate((J, h))
np.savetxt(f"../data/hidden_nodes/Jh_questions_{n}_samples_{C}_scale_{scale}.dat", Jh)

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
         f'../data/hidden_nodes/questions_{n}_samples_{C}_scale_{scale}_hidden_NONE.dat')

# save all n-1 
samplex = np.copy(sample)
samplex[:, 0] = 0
samplex
_, columns = sample.shape
for i in range(columns): 
    sample_i = np.copy(sample)
    sample_i[:, i] = 0 
    bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in sample_i]
    save_dat(bit_string, weight_string, sample_i,
             f'../data/hidden_nodes/questions_{n-1}_samples_{C}_scale_{scale}_hidden_{i}.dat')

sample_i[0, i] = 1
sample_i
bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in sample_i]
save_dat(bit_string, weight_string, sample_i,
            f'../data/hidden_nodes/questions_{n-1}_samples_{C}_scale_{scale}_hidden_{i}_test.dat')

# knock out one node 
# save that data 