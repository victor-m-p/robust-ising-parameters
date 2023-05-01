'''
VMP 2023-04-30:
Testing the correlation we get with different types of distributions
over the parameters. 
'''

import numpy as np 
from sample_functions import sample_not_connected, sample_hidden_connected, sample_fully_connected, save_to_mpf
import matplotlib.pyplot as plt 
import seaborn as sns 

# for saving to mpf format
conversion_dict = {
    '-1': '0',
    '0': 'X',
    '1': '1'
}

# meta settings
outpath_true = 'data/not_connected_true/'
outpath_mpf = 'data/not_connected_mpf/'
h_type = 'gaussian01'
J_type = 'gaussian01'

# overall params (maximum 11 nodes) 
n_hidden = 2 
n_visible = 4
list_hidden_implied = [0, 2, 4]
n_simulations = 500 

# loop over different combinations 

# set up data for the independent model
h_hidden = np.random.normal(0, 1, n_hidden) 
h_visible = np.random.normal(0, 1, n_visible)  
J_interlayer = np.random.normal(0, 1, (n_visible, n_hidden))

# save the true parameters in mpf compatible format 
# including the implied parameters that we do not observe?
## reformat h 
h = np.concatenate((h_hidden, h_visible)) 

## reformat J 
J_interlayer_flat = J_interlayer.flatten(order='F') # flatten column-major style
n_nodes = n_hidden + n_visible
J_hidden = np.zeros(int(n_hidden*(n_hidden-1)/2))
J_visible = np.zeros(int(n_visible*(n_visible-1)/2))
J = np.concatenate((J_hidden, J_interlayer_flat, J_visible)) 

## concatenate mpf style 
Jh = np.concatenate((J, h))
np.savetxt(f'{outpath_true}format_Jh_nhid_{n_hidden}_nvis_{n_visible}_th_{h_type}_tj_{J_type}.txt',
           Jh) 

# sample from model that is independent in both layers
sim_not_connected = sample_not_connected(
    n_simulations, 
    h_hidden, 
    h_visible, 
    J_interlayer
    )

# this we save both in regular and mpf compatible format  
np.savetxt(f'{outpath_true}sim_true_nhid{n_hidden}_nvis_{n_visible}_th_{h_type}_tj_{J_type}.txt',
           sim_not_connected,
           fmt='%d') 

save_to_mpf(sim_not_connected,
            conversion_dict,
            f'{outpath_mpf}sim_vis_mpf_nhid_{n_hidden}_nvis_{n_visible}_th_{h_type}_tj_{J_type}.txt')

# now for different number of implied hidden nodes 
sim_visible = sim_not_connected[:, n_hidden:]

# loop over different "proposals" for number of hidden nodes
for i in list_hidden_implied: 
    # imply a certain number of hidden nodes
    sample_i_implied = np.concatenate((np.zeros((n_simulations, i)), sim_visible), axis=1)
    
    # reformat to mpf compatible format 
    save_to_mpf(sample_i_implied, 
                conversion_dict, 
                f'{outpath_mpf}sim_hid_mpf_nhid_{i}_nvis_{n_visible}_th_{h_type}_tj_{J_type}.txt')