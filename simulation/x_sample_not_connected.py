'''
VMP 2023-04-30:
Testing the correlation we get with different types of distributions
over the parameters. 
'''

import numpy as np 
from sample_functions import sample_not_connected, construct_J, save_to_mpf
import os 

# for saving to mpf format
conversion_dict = {
    '-1': '0',
    '0': 'X',
    '1': '1'
}


# meta settings 
h_type = 'gaussian'
J_type = 'gaussian'
h_mean = 0.0
h_std = 0.5
J_mean = 0.0
J_std = 0.5

# overall params 
n_visible = 10
n_hidden = 1
n_nodes = n_hidden + n_visible
list_hidden_implied = [n_hidden]
n_sim = 500 
norm = 'l1'

# create directory if does not exist
outpath_mpf = f"data/not_connected_nn{n_nodes}_nsim{n_sim}_{norm}_mpf/"
outpath_true = f'data/not_connected_nn{n_nodes}_nsim{n_sim}_true/'

if not os.path.exists(outpath_mpf): 
    os.makedirs(outpath_mpf)

if not os.path.exists(outpath_true):
    os.makedirs(outpath_true)

# set up data for the independent model
h_hidden = np.random.normal(h_mean, h_std, n_hidden) 
h_visible = np.random.normal(h_mean, h_std, n_visible)  
J_interlayer = np.random.normal(J_mean, J_std, (n_visible, n_hidden))

# save the true parameters in mpf compatible format 
# including the implied parameters that we do not observe?
## reformat h 
h = np.concatenate((h_hidden, h_visible)) 

## reformat J (are we sure this is right?)
## I don't think this is correct actually  
J_interlayer_flat = J_interlayer.flatten(order='F') # flatten column-major style
n_nodes = n_hidden + n_visible
J_hidden = np.zeros(int(n_hidden*(n_hidden-1)/2))
J_visible = np.zeros(int(n_visible*(n_visible-1)/2))
J = construct_J(J_hidden, J_interlayer_flat, J_visible, n_hidden, n_visible) 

## concatenate mpf style 
Jh = np.concatenate((J, h))
np.savetxt(f'{outpath_true}format_Jh_nhid_0_nvis_{n_nodes}_th_{h_type}_{h_mean}_{h_std}_tj_{J_type}_{J_mean}_{J_std}_nsim_{n_sim}.txt',
           Jh) 

sim = sample_not_connected(
    n_sim, 
    h_hidden, 
    h_visible, 
    J_interlayer
    )

# this we save in regular format and mpf format 
np.savetxt(f'{outpath_true}sim_true_nhid_0_nvis_{n_nodes}_th_{h_type}_{h_mean}_{h_std}_tj_{J_type}_{J_mean}_{J_std}_nsim_{n_sim}.txt',
           sim,
           fmt='%d') 

save_to_mpf(sim,
            conversion_dict,
            f'{outpath_true}sim_mpf_nhid_0_nvis_{n_nodes}_th_{h_type}_{h_mean}_{h_std}_tj_{J_type}_{J_mean}_{J_std}_nsim_{n_sim}.txt')

# now for different number of implied hidden nodes 
n_visible=n_nodes-n_hidden
sim_visible = sim[:, n_hidden:]

save_to_mpf(sim_visible,
            conversion_dict,
            f'{outpath_mpf}sim_mpf_nhid_0_nvis_{n_visible}_th_{h_type}_{h_mean}_{h_std}_tj_{J_type}_{J_mean}_{J_std}_nsim_{n_sim}.txt')

# loop over different "proposals" for number of hidden nodes
for i in list_hidden_implied:
    # imply a certain number of hidden nodes
    sample_i_implied = np.concatenate((np.zeros((n_sim, i)), sim_visible), axis=1)
    
    # reformat to mpf compatible format 
    save_to_mpf(sample_i_implied, 
                conversion_dict, 
                f'{outpath_mpf}sim_mpf_nhid_{i}_nvis_{n_visible}_th_{h_type}_{h_mean}_{h_std}_tj_{J_type}_{J_mean}_{J_std}_nsim_{n_sim}.txt')