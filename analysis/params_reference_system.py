'''
VMP 2023-02-10: get reference hi and Jij for 20 nodes
'''

import itertools 
import pandas as pd 
import numpy as np 

def get_hi_Jij(n, corr_J, means_h):
    nodes = range(1, n+1)
    Jij = pd.DataFrame(list(itertools.combinations(nodes, 2)), columns=['i', 'j'])
    Jij['coupling'] = corr_J
    hi = pd.DataFrame({'q': nodes, 'h': means_h})
    return hi, Jij

# setup 
n_nodes, n_nan, n_rows, n_entries = 20, 5, 455, 407
A = np.loadtxt(f'../data/mdl_experiments/matrix_questions_{n_nodes}_maxna_{n_nan}_nrows_{n_rows}_entries_{n_entries}.txt.mpf_params.dat')
n_J = int(n_nodes*(n_nodes-1)/2)
J = A[:n_J] 
h = A[n_J:]

# run and save 
hi_20, Jij_20 = get_hi_Jij(n_nodes, J, h)
hi_20.to_csv('../data/analysis/hi_20.csv', index=False)
Jij_20.to_csv('../data/analysis/Jij_20.csv', index=False)