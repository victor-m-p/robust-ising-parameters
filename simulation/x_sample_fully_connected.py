'''
VMP 2023-04-30:
Testing the correlation we get with different types of distributions
over the parameters. 
'''

import numpy as np 
from sample_functions import sample_fully_connected, save_to_mpf
import os 

# for saving to mpf format
conversion_dict = {
    '-1': '0',
    '0': 'X',
    '1': '1'
}

# overall params (maximum 11 nodes) 
n_nodes = 20
n_hidden = 15
list_hidden_implied = [1, 2, 3]
n_connections = int(n_nodes*(n_nodes-1)/2)
n_sim = 500

# meta settings
outpath_true = f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_true/'
outpath_mpf = f'data/fully_connected_nn{n_nodes}_nsim{n_sim}_mpf/'
os.mkdir(f'{outpath_true}')
os.mkdir(f'{outpath_mpf}')
h_type = 'gaussian'
J_type = 'gaussian'
h_mean = 0.0
h_std = 0.1
J_mean = 0.0
J_std = 0.1

# loop over different combinations 

# set up data for the independent model
h = np.random.normal(h_mean, h_std, n_nodes)  
J = np.random.normal(J_mean, J_std, n_connections)

## concatenate mpf style 
Jh = np.concatenate((J, h))
np.savetxt(f'{outpath_true}format_Jh_nhid_0_nvis_{n_nodes}_th_{h_type}_{h_mean}_{h_std}_tj_{J_type}_{J_mean}_{J_std}_nsim_{n_sim}.txt',
           Jh) 

# sample from model that is independent in both layers
sim = sample_fully_connected(
    n_sim, 
    h,
    J)

# this we save both in regular and mpf compatible format  
np.savetxt(f'{outpath_true}sim_true_nhid_0_nvis_{n_nodes}_th_{h_type}_{h_mean}_{h_std}_tj_{J_type}_{J_mean}_{J_std}_nsim_{n_sim}.txt',
           sim,
           fmt='%d') 

#save_to_mpf(sim,
#            conversion_dict,
#            f'{outpath_mpf}sim_hid_mpf_nhid_0_nvis_{n_nodes}_th_{h_type}_{h_mean}_{h_std}_tj_{J_type}_{J_mean}_{J_std}_nsim_{n_sim}.txt')

# now for different number of implied hidden nodes 
n_visible=n_nodes-n_hidden
sim_visible = sim[:, n_hidden:]

# loop over different "proposals" for number of hidden nodes
for i in list_hidden_implied:
    # imply a certain number of hidden nodes
    sample_i_implied = np.concatenate((np.zeros((n_sim, i)), sim_visible), axis=1)
    
    # reformat to mpf compatible format 
    save_to_mpf(sample_i_implied, 
                conversion_dict, 
                f'{outpath_mpf}sim_hid_mpf_nhid_{i}_nvis_{n_visible}_th_{h_type}_{h_mean}_{h_std}_tj_{J_type}_{J_mean}_{J_std}_nsim_{n_sim}.txt')
