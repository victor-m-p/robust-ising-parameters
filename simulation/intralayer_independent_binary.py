'''
VMP 2023-04-28: 
The simplest case. 
(1) only intralayer connections
(2) all layers are binary
(3) hidden features independent
'''

import numpy as np 

# 1. the simplest case: 
n_sample = 10000 # all independent (no repeated measurement)
n_visible = 3 # number of visible units
n_hidden = 1 # number of hidden units 
weight_distribution = np.random.normal(0, 1, (n_visible, n_hidden)) # weight from hidden to visible layer
bias_distribution = np.random.uniform(0, 1, n_hidden) # probability of hidden layer
samp_visible = np.zeros((n_sample, n_visible)) # init samples of visible units
for i in range(n_sample): 
    error_sample = np.random.normal(0, 1, (n_visible, n_hidden)) # error term
    weight_sample = weight_distribution + error_sample # error on weight 
    bias_sample = np.random.binomial(1, bias_distribution, n_hidden) # bias of hidden layer
    samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(bias_sample, weight_sample.T))))) # yi

# we basically get same base rate + no correlation 
np.mean(samp_visible, axis=0) 
np.corrcoef(samp_visible, rowvar=False) 
np.cov(samp_visible.T) # uncorrelated 

## things we can vary here: 
# * number visible nodes
# * number hidden nodes
# * weight distribution
# * error distribution
# * bias distribution 

# a function we can run over some grid of parameters 
def intralayer_independent_binary(n_sample,
                                 n_visible,
                                 n_hidden,
                                 weight_params = (0, 1),
                                 error_params = (0, 1),
                                 bias_params = (0, 1)):
    w_mean, w_sd = weight_params
    e_mean, e_sd = error_params
    b_lower, b_upper = bias_params
    weight_distribution = np.random.normal(w_mean, w_sd, (n_visible, n_hidden)) 
    bias_distribution = np.random.uniform(b_lower, b_upper, n_hidden)
    samp_visible = np.zeros((n_sample, n_visible))
    for i in range(n_sample): 
        error_sample = np.random.normal(e_mean, e_sd, (n_visible, n_hidden)) 
        weight_sample = weight_distribution + error_sample
        bias_sample = np.random.binomial(1, bias_distribution, n_hidden)
        samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(bias_sample, weight_sample.T)))))
    return samp_visible 

# run over grid and plot some summary statistics for mean and correlation
# basically we want the spread of what we get from this at different values 
n_sample = 1000
n_visible = 5
n_hidden = 3 

weight_params = (0, 1)
error_params = (0, 1)
bias_params = (0, 1)

sample = intralayer_independent_binary(n_sample,
                                       n_visible,
                                       n_hidden,
                                       weight_params,
                                       error_params,
                                       bias_params)

np.corrcoef(sample, rowvar=False)
np.mean(sample, axis=0) # can get nan if one is always 0 or 1. 
# higher weight = higher correlation (but never really high correlation).
