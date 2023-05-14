'''
this can be deleted soon, because validation is done. 
'''

import numpy as np 
import itertools
from sample_functions import bin_states, ising_probs

# compare to probabilities from mpf code 
def recode_mpf_format(p_mpf, n_nodes): 
    python_format = bin_states(n_nodes)
    mpf_format = python_format[:, ::-1]
    idx_recode = np.array([np.where(np.all(mpf_format == i, axis=1))[0][0] for i in python_format])
    return p_mpf[idx_recode]

# meta params
n_nodes = 11
n_connections = int(n_nodes*(n_nodes-1)/2)

# load the mpf 
mpf_p = np.loadtxt('/home/vmp/robust-ising-parameters/simulation/data/mpf_sanity/format_Jh_nhid_0_nvis_11_th_gaussian_0.0_0.1_tj_gaussian_0.0_0.1_nsim_5000.txt_probs.dat')
mpf_p = mpf_p[:, 2] # the probabilities I think
mpf_p_recoded = recode_mpf_format(mpf_p, n_nodes)

# load the true
# why do I take out h first?
# I should be taking J out first. 
true_params = np.loadtxt('/home/vmp/robust-ising-parameters/simulation/data/mpf_sanity/format_Jh_nhid_0_nvis_11_th_gaussian_0.0_0.1_tj_gaussian_0.0_0.1_nsim_5000.txt')
h_true = true_params[n_connections:]
J_true = true_params[:n_connections]

# run forward 
p_true = ising_probs(h_true, J_true)

# our probabilities are the same 
# but our log likelihood is not the same 

# is it "relatively" the same? 
